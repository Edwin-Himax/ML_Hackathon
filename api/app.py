"""Run: .venv/Scripts/python -m uvicorn api.app:app --host 0.0.0.0 --port 8000"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import threading
import time
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from pathsnap import __version__
from pathsnap.settings import Settings, ROOT
from pathsnap.pipeline import Predictor, public_prediction
from pathsnap.evaluation import Scorer

logger = logging.getLogger('uvicorn.error')


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    trip_id: str = Field(pattern=r'^\d{8}$')


def create_app(settings=None):
    settings = settings or Settings()
    prediction_lock = threading.Lock()
    log_lock = threading.Lock()

    def log(record):
        record['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(record)
        logger.info('PATHSNAP %s', line)
        settings.log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_lock, settings.log_path.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = Predictor(settings.bundle_path, settings.decoder_config())
        app.state.scorer = Scorer(settings.reference_root, settings.pointwise_root)
        yield

    app = FastAPI(title='PathSnap', version=__version__, lifespan=lifespan,
                  description='GPS road matching with classifier and directed-graph Viterbi decoding. Reference scoring is isolated and aggregate-only.')
    app.state.settings = settings
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=False, allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

    def trip_list():
        root = settings.partition_root
        if not root.is_dir():
            return []
        return [p for p in sorted(root.iterdir()) if p.is_dir() and re.fullmatch(r'\d{8}', p.name) and all((p / f'{p.name}.{ext}').is_file() for ext in ['track', 'nodes', 'arcs'])]

    @app.get('/health')
    def health(request: Request):
        trips = trip_list()
        predictor = request.app.state.predictor
        return {'status': 'ready' if trips else 'waiting_for_inputs', 'version': __version__, 'model': predictor.bundle['model_name'], 'model_version': predictor.model_version,
                'partition': settings.partition, 'trip_count': len(trips), 'mode': 'Final evaluation' if settings.partition == 'evaluator_test' else 'Development',
                'reference_policy': 'server_side_aggregate_only', 'classifier_label_status': predictor.bundle['label_status'],
                'decoder': predictor.decoder_config}

    @app.get('/trips')
    def trips():
        entries = []
        for folder in trip_list():
            # Input observations only. No route-file access for listing.
            with (folder / f'{folder.name}.track').open() as stream:
                observations = sum(1 for line in stream if line.strip())
            entries.append({'trip_id': folder.name, 'label': f'Trip {int(folder.name):02d}', 'observations': observations})
        return {'partition': settings.partition, 'trips': entries, 'notice': 'Reserved evaluator inputs are not included in the supplied dataset.' if not entries and settings.partition == 'evaluator_test' else None}

    @app.post('/predict')
    def predict(body: PredictRequest, request: Request):
        request_id = uuid.uuid4().hex[:12]
        allowed = {p.name: p for p in trip_list()}
        if body.trip_id not in allowed:
            log({'event': 'prediction_rejected', 'request_id': request_id, 'trip_id': body.trip_id, 'status': 404})
            raise HTTPException(status_code=404, detail='Trip not found in the active partition')
        start = time.perf_counter()
        try:
            with prediction_lock:
                prediction = request.app.state.predictor.predict(allowed[body.trip_id])
                # Prediction is complete before any reference data is opened.
                score_start = time.perf_counter()
                try:
                    metrics = request.app.state.scorer.score(prediction, request.app.state.predictor.bundle)
                    scorer_warning = None
                except (ValueError, OSError) as exc:
                    logger.warning('Scoring unavailable for request %s: %s', request_id, type(exc).__name__)
                    metrics = {'status': 'unavailable', 'label_notice': 'Reference scoring is unavailable; the predicted route is still valid.', 'eligible_observations': 0, 'excluded_observations': len(prediction.trip.track), 'classifier': {}, 'hmm': {}, 'candidate_miss_rate': None}
                    scorer_warning = 'Reference scoring unavailable'
                response = public_prediction(prediction)
                response['timings']['scoring_s'] = time.perf_counter() - score_start
                response.update({'request_id': request_id, 'partition': settings.partition, 'evaluation': metrics,
                                 'warnings': ([metrics['label_notice']] if metrics['status'] != 'official_pointwise' else []) + ([scorer_warning] if scorer_warning else [])})
                response['timings']['request_s'] = time.perf_counter() - start
            log({'event': 'prediction_complete', 'request_id': request_id, 'trip_id': body.trip_id, 'partition': settings.partition,
                 'model_version': prediction.model_version, 'observations': len(prediction.trip.track), 'hmm_resets': len(response['hmm']['resets']),
                 'evaluation_status': metrics['status'], 'timings': response['timings'], 'status': 200})
            return response
        except (ValueError, OSError) as exc:
            log({'event': 'prediction_failed', 'request_id': request_id, 'trip_id': body.trip_id, 'error_type': type(exc).__name__, 'status': 422})
            raise HTTPException(status_code=422, detail='Trip inputs could not be processed. See the server log and validate the dataset.') from exc

    @app.get('/', include_in_schema=False)
    def index():
        return RedirectResponse('/ui/')

    app.mount('/ui', StaticFiles(directory=ROOT / 'ui', check_dir=False, html=True), name='ui')
    return app


app = create_app()
