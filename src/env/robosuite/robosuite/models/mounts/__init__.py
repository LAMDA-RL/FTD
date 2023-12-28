from .mount_factory import mount_factory
from .mount_model import MountModel
from .null_mount import NullMount
from .rethink_minimal_mount import RethinkMinimalMount
from .rethink_mount import RethinkMount

MOUNT_MAPPING = {
    "RethinkMount": RethinkMount,
    "RethinkMinimalMount": RethinkMinimalMount,
    None: NullMount,
}

ALL_MOUNTS = MOUNT_MAPPING.keys()
