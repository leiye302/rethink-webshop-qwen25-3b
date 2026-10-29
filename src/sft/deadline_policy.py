"""User-authorized runtime shutdown control, separate from scientific config."""
from datetime import datetime
import os
import time


class EvaluationDeadline(RuntimeError):
    pass


def stop_epoch():
    value=os.environ.get('WEBSHOP_SAVE_BY_ISO')
    if not value: return None
    parsed=datetime.fromisoformat(value)
    if parsed.tzinfo is None: raise ValueError('Shutdown protection requires an explicit timezone')
    return parsed.timestamp()


def policy():
    return {'save_by':os.environ.get('WEBSHOP_SAVE_BY_ISO'),
            'server_shutdown':os.environ.get('WEBSHOP_SHUTDOWN_AT_ISO'),
            'training_seconds_per_update_budget':4.5,'checkpoint_reserve_seconds':300}


def evaluation_cutoff(completed,total=1500):
    deadline=stop_epoch()
    return None if deadline is None else deadline-max(0,total-completed)*4.5-300


def check_evaluation(cfg):
    deadline=cfg.get('_evaluation_deadline_epoch')
    if deadline is not None and time.time()>=deadline:
        raise EvaluationDeadline('Evaluation deferred to preserve training/checkpoint time before shutdown')
