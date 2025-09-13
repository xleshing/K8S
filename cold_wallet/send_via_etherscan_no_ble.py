# -*- coding: utf-8 -*-
# 離線簽名（手機寫 FFF3、取 FFF4 DER）→ 反解地址 → (可選) 廣播到 Sepolia
import sys, binascii, requests, rlp
from typing import Tuple
from decimal import Decimal
from Crypto.Hash import keccak as _keccak
from eth_keys import keys
from eth_utils import to_checksum_address

CHAIN_ID = 1#1155111  # Sepolia
ETHERSCAN = "https://api.etherscan.io/v2/api"

# <<< 在這裡寫死你的 API Key >>>
ETHERSCAN_API_KEY = "C2V3NT7U339XK5P4H3IU22RQZ5WXF7717S"

# ---------------- Keccak-256 ----------------
def keccak256(data: bytes) -> bytes:
    k = _keccak.new(digest_bits=256)
    k.update(data)
    return k.digest()

# ---------------- RLP / EIP-1559 helpers ----------------
def int_to_be(x: int) -> bytes:
    # 嚴格保證回傳 bytes，不會是 None
    if not isinstance(x, int):
        raise TypeError(f"int_to_be expects int, got {type(x)}")
    if x == 0:
        return b""
    return x.to_bytes((x.bit_length() + 7) // 8, "big")

def hex_to_bytes(x: str) -> bytes:
    x = x.strip();  x = x[2:] if x.lower().startswith("0x") else x
    if len(x) % 2: x = "0"+x
    return binascii.unhexlify(x)

def eip1559_signing_hash(chain_id:int, nonce:int, max_prio:int, max_fee:int,
                         gas:int, to:str, value:int, data:bytes=b"", access_list=None) -> bytes:
    if access_list is None:
        access_list = []  # 空 accessList

    # 類型檢查（除錯用）
    dbg = {
        "chain_id": (chain_id, type(chain_id)),
        "nonce": (nonce, type(nonce)),
        "max_prio": (max_prio, type(max_prio)),
        "max_fee": (max_fee, type(max_fee)),
        "gas": (gas, type(gas)),
        "to": (to, type(to)),
        "value": (value, type(value)),
        "data": (data, type(data)),
        "access_list": (access_list, type(access_list)),
    }

    to_bytes = b"" if (to is None or to == "" or to == "0x" or to == "0X") else hex_to_bytes(to)

    try:
        payload = [
            int_to_be(int(chain_id)),
            int_to_be(int(nonce)),
            int_to_be(int(max_prio)),
            int_to_be(int(max_fee)),
            int_to_be(int(gas)),
            to_bytes,
            int_to_be(int(value)),
            data if isinstance(data, (bytes, bytearray)) else bytes(data),
            []  # 一律空 accessList
        ]
        encoded = b"\x02" + rlp.encode(payload)
    except Exception as e:
        print("RLP 參數型別檢查：")
        for k, (v, t) in dbg.items():
            print(f"  {k}: {v!r}  ({t})")
        raise
    return keccak256(encoded)

def eip1559_serialize_signed(chain_id, nonce, max_prio, max_fee, gas, to, value, data, r, s, y_parity):
    to_bytes = b"" if not to else hex_to_bytes(to)
    payload = [int_to_be(chain_id), int_to_be(nonce), int_to_be(max_prio), int_to_be(max_fee),
               int_to_be(gas), to_bytes, int_to_be(value), data, [],
               int_to_be(y_parity), int_to_be(r), int_to_be(s)]
    return "0x" + (b"\x02" + rlp.encode(payload)).hex()

# ---------------- DER 解析（含 long-form length）+ low-S ----------------
SECP256K1_N  = int("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16)
SECP256K1_NH = SECP256K1_N // 2

def _read_len(b: bytes, i: int):
    L = b[i]; i += 1
    if L & 0x80:
        n = L & 0x7F
        if n == 0 or n > 4:
            raise ValueError("DER length-of-length too big")
        L = int.from_bytes(b[i:i+n], "big"); i += n
    return L, i

def parse_der_rs(der_hex: str):
    """回傳 (r, s, flipped)。若做了 low-S 規範化，flipped=True（需翻轉 parity）。"""
    b = hex_to_bytes(der_hex)
    i = 0
    if len(b) < 8:
        raise ValueError("DER too short")

    if b[i] != 0x30:
        raise ValueError(f"DER: not SEQUENCE (got 0x{b[i]:02x})")
    i += 1

    seq_len, i = _read_len(b, i)
    end = i + seq_len
    if end > len(b):
        raise ValueError("DER: length exceeds buffer")

    if b[i] != 0x02:
        raise ValueError(f"DER: r tag (got 0x{b[i]:02x} at {i})")
    i += 1
    rlen, i = _read_len(b, i)
    if i + rlen > end:
        raise ValueError("DER: r length overflow")
    r_bytes = b[i:i+rlen]; i += rlen

    if i >= end or b[i] != 0x02:
        got = b[i] if i < len(b) else None
        raise ValueError(f"DER: s tag (got {None if got is None else hex(got)} at {i})")
    i += 1
    slen, i = _read_len(b, i)
    if i + slen > end:
        raise ValueError("DER: s length overflow")
    s_bytes = b[i:i+slen]; i += slen

    # 允許尾端多餘 0 長度（通常不會有）
    # if i != end: pass

    r = int.from_bytes(r_bytes, "big")
    s = int.from_bytes(s_bytes, "big")

    flipped = False
    if s > SECP256K1_NH:
        s = SECP256K1_N - s
        flipped = True

    return r, s, flipped

# ---------------- Etherscan Proxy API ----------------
def api_get(action: str, **params):
    q = {"chainid": CHAIN_ID, "module":"proxy", "action":action, "apikey": ETHERSCAN_API_KEY}
    q.update(params)
    j = requests.get(ETHERSCAN, params=q, timeout=30).json()
    if "error" in j: raise RuntimeError(j["error"])
    return j["result"]

def get_nonce(addr: str) -> int:
    return int(api_get("eth_getTransactionCount", address=addr, tag="latest"), 16)

def get_balance_eth(addr: str) -> float:
    j = requests.get(ETHERSCAN, params={
        "chainid": CHAIN_ID, "module":"account", "action":"balance",
        "address": addr, "tag":"latest", "apikey": ETHERSCAN_API_KEY
    }, timeout=20).json()
    return int(j["result"]) / 1e18

# ---------------- 工具 ----------------
def eth_to_wei(x: str) -> int:  return int(Decimal(x) * Decimal(10**18))
def gwei_to_wei(x: str) -> int: return int(Decimal(x) * Decimal(10**9))

def recover_address(digest32: bytes, r: int, s: int, y_parity: int) -> str:
    # eth_keys 要的 v 是 0 或 1，不是 27/28
    sig = keys.Signature(vrs=(y_parity, r, s))
    pub = sig.recover_public_key_from_msg_hash(digest32)
    return pub.to_checksum_address()

# ---------------- 主流程 ----------------
def main():
    print("=== Sepolia：離線簽名（FFF3/FFF4）→ 反解 sender → (可選) 廣播 ===")

    from_addr = "0x4CBABB70E70463254703CB656AAAFF11E93C67BD".strip()
    to_addr   = "0x26DB6a712438980Bde4088c0746523b5d8ccC23B".strip()
    amount    = "0.000026"
    max_fee_gwei  = "1"
    prio_gwei     = "1"
    gas_limit     = "21000"

    bal = get_balance_eth(from_addr)
    print(f"From 餘額：{bal:.6f} ETH")

    try:
        nonce = get_nonce(from_addr)
    except Exception as e:
        print("取 nonce 失敗：", e); sys.exit(1)

    value = eth_to_wei(amount)
    max_fee  = gwei_to_wei(max_fee_gwei)
    max_prio = gwei_to_wei(prio_gwei)
    gas      = int(gas_limit)
    data     = b""

    digest = eip1559_signing_hash(CHAIN_ID, nonce, max_prio, max_fee, gas, to_addr, value, data)
    print("\n➡️ 對 FFF3 寫入 32B digest（Hex）:")
    print("0x" + digest.hex())
    print("完成後，貼 FFF4 DER 簽名。")

    der_hex = input("\nFFF4 DER（0x... 或 hex）: ").strip()
    try:
        r, s, flipped = parse_der_rs(der_hex)
    except Exception as e:
        print("DER 解析失敗：", e); sys.exit(1)

    print(f"\nr = {hex(r)}"); print(f"s = {hex(s)}"); print(f"low-S flipped = {flipped}")

    candidates = [0, 1] if not flipped else [1, 0]
    chosen = None; recovered_addr = None
    for y in candidates:
        try:
            addr = recover_address(digest, r, s, y)  # 直接給 0/1
        except Exception as e:
            print(f"y_parity={y} 反解失敗：{e}")
            continue
        print(f"y_parity={y} 反解地址：{addr}")
        if addr.lower() == from_addr.lower():
            chosen = y
            recovered_addr = addr
            break

    if chosen is None:
        print("\n❗簽名反解出的地址與 From 不符。請檢查 digest/DER 是否對應同一筆。")
        sys.exit(1)

    print(f"\n✅ 使用 y_parity={chosen}（v={27+chosen}）。Sender 驗證通過：{recovered_addr}")

    raw = eip1559_serialize_signed(CHAIN_ID, nonce, max_prio, max_fee, gas, to_addr, value, data, r, s, chosen)
    go = input("\n是否廣播到 Sepolia？(y/N): ").strip().lower() == "y"
    if not go:
        print("\n已產生 raw tx（未送出）:\n", raw); return

    res = requests.get(ETHERSCAN, params={
        "chainid": CHAIN_ID, "module":"proxy", "action":"eth_sendRawTransaction",
        "hex": raw, "apikey": ETHERSCAN_API_KEY
    }, timeout=60).json()
    print("\n結果：", res)
    if isinstance(res, dict) and isinstance(res.get("result"), str) and res["result"].startswith("0x"):
        print("🎉 成功！tx hash =", res["result"])
        print(f"查看： https://sepolia.etherscan.io/tx/{res['result']}")

if __name__ == "__main__":
    main()
