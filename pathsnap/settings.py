from dataclasses import dataclass, field
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    data_root: Path = field(default_factory=lambda: Path(os.environ.get('PATHSNAP_DATA_ROOT', ROOT / 'Student folder' / 'Student folder')))
    partition: str = field(default_factory=lambda: os.environ.get('PATHSNAP_PARTITION', 'student_test'))
    bundle_path: Path = field(default_factory=lambda: Path(os.environ.get('PATHSNAP_MODEL', ROOT / 'artifacts' / 'member2' / 'final_model.joblib')))
    decoder_path: Path = field(default_factory=lambda: Path(os.environ.get('PATHSNAP_DECODER', ROOT / 'configs' / 'decoder.json')))
    reference_root: Path | None = None
    pointwise_root: Path | None = None
    allowed_origins: list = field(default_factory=lambda: os.environ.get('PATHSNAP_ALLOWED_ORIGINS', 'http://localhost:8001,http://127.0.0.1:8001').split(','))
    log_path: Path = field(default_factory=lambda: Path(os.environ.get('PATHSNAP_LOG', ROOT / 'runtime' / 'api_requests.jsonl')))

    def __post_init__(self):
        if self.partition not in ['train', 'student_test', 'evaluator_test']:
            raise ValueError('Partition must be train, student_test or evaluator_test')
        if self.reference_root is None and os.environ.get('PATHSNAP_REFERENCE_ROOT'):
            self.reference_root = Path(os.environ['PATHSNAP_REFERENCE_ROOT'])
        if self.pointwise_root is None and os.environ.get('PATHSNAP_POINTWISE_ROOT'):
            self.pointwise_root = Path(os.environ['PATHSNAP_POINTWISE_ROOT'])
        # Only development partitions may automatically locate their supplied references.
        if self.reference_root is None and self.partition != 'evaluator_test':
            self.reference_root = self.data_root / self.partition

    @property
    def partition_root(self):
        return self.data_root / self.partition

    def decoder_config(self):
        cfg = json.loads(self.decoder_path.read_text(encoding='utf-8'))
        required_positive = ['transition_beta_m', 'emission_weight', 'gap_reset_s', 'max_speed_mps', 'maximum_path_m']
        if any(float(cfg[k]) <= 0 for k in required_positive):
            raise ValueError('Invalid decoder parameters')
        if any(float(cfg[k]) < 0 for k in ['u_turn_penalty', 'distance_slack_m', 'backward_jitter_m']):
            raise ValueError('Invalid decoder bounds')
        return cfg
