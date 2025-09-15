import os
from flask import Flask, request, jsonify
from lstm_model import (
    get_pod_data,
    prepare_data,
    train_model,
    predict_at_time,
    init_model_and_optim,
    load_model,
    load_scaler,
    get_connection,
)
app = Flask(__name__)

@app.route('/predict_time', methods=['GET'])
def predict_time():
    pod = request.args.get('pod')
    target_time = request.args.get('time')  # e.g. 2025-05-28 12:00
    if not pod or '/' not in pod:
        return jsonify({'error': 'Invalid pod format. Use namespace/pod_name'}), 400
    if not target_time:
        return jsonify({'error': 'Missing time query parameter (e.g. 2025-05-28 12:00)'}), 400

    namespace, pod_name = pod.split('/', 1)
    df = get_pod_data(namespace, pod_name)
    if df is None or len(df) < 31:
        return jsonify({'error': 'Not enough data to make prediction'}), 400

    model_path = os.path.join('/models', f"{namespace}_{pod_name}.pt")
    scaler_path = os.path.join('/models', f"{namespace}_{pod_name}_scaler.pkl")
    model = None
    scaler = None

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model, optim, _ = init_model_and_optim(input_size=6, hidden_size=64, num_layers=2)
            load_model(model, optim, model_path)
            scaler = load_scaler(scaler_path)
        except Exception:
            model = None
            scaler = None

    if model is None or scaler is None:
        X, y, scaler = prepare_data(df)
        model = train_model(X, y)

    try:
        prediction = predict_at_time(model, df, target_time, scaler)
    except Exception as e:
        return jsonify({'error': f'Prediction error: {e}'}), 500

    return jsonify({'pod': pod, 'predicted_cpu_usage_at': target_time, 'cpu': prediction})

@app.route('/history', methods=['GET'])
def history():
    pod = request.args.get('pod')
    if not pod or '/' not in pod:
        return jsonify({'error': 'Invalid pod format. Use namespace/pod_name'}), 400
    namespace, pod_name = pod.split('/', 1)
    df = get_pod_data(namespace, pod_name)
    return df.to_json(orient='records')

def _fetch_all_pods():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT pod_namespace, pod_name FROM pod_cpu_usage")
            rows = cur.fetchall()
        return [(r['pod_namespace'], r['pod_name']) for r in rows]
    finally:
        conn.close()

@app.route('/predict_cluster', methods=['GET'])
def predict_cluster():
    target_time = request.args.get('time')
    if not target_time:
        return jsonify({'error': 'Missing time query parameter (e.g. 2025-05-28 12:00)'}), 400

    pods = _fetch_all_pods()
    if not pods:
        return jsonify({'error': 'No pods found in database'}), 400

    predictions, total_cpu = [], 0.0
    for namespace, pod_name in pods:
        df = get_pod_data(namespace, pod_name)
        if df is None or len(df) < 31:
            continue
        model_path = os.path.join('/models', f"{namespace}_{pod_name}.pt")
        scaler_path = os.path.join('/models', f"{namespace}_{pod_name}_scaler.pkl")
        model = None
        scaler = None
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                m, optim, _ = init_model_and_optim(input_size=6, hidden_size=64, num_layers=2)
                load_model(m, optim, model_path)
                model = m
                scaler = load_scaler(scaler_path)
            except Exception:
                model = None
                scaler = None
        if model is None or scaler is None:
            X, y, scaler = prepare_data(df)
            model = train_model(X, y)
        try:
            pred = predict_at_time(model, df, target_time, scaler)
            total_cpu += pred
            predictions.append({'pod': f'{namespace}/{pod_name}', 'cpu': pred})
        except Exception:
            continue

    return jsonify({'time': target_time, 'total_cpu': total_cpu, 'details': predictions})

if __name__ == '__main__':
    # dev 模式用；k8s 用 gunicorn，見 Dockerfile.api
    app.run(host='0.0.0.0', port=5000)