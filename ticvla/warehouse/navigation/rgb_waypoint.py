"""Source-bound RGB pixel waypoints. Never reinterpret legacy metric labels.

The fitted curve is an engineering controller input, NOT Reference Decoder output.
Depth is optical Z in metres, pixels use the original image (u right, v down).
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import PchipInterpolator
from ..geometry import transform_points
from ..schema import file_sha256

PROTOCOL = 'rgb-pixel-waypoint-v1'


def project_waypoints(pixels, source, *, check_hashes=True):
    if check_hashes:
        for field in ('rgb', 'depth'):
            if file_sha256(source[field]) != source[field + '_hash']:
                raise ValueError('SOURCE_HASH_MISMATCH:' + field)
    depth = np.load(source['depth'], allow_pickle=False)
    from PIL import Image
    with Image.open(source['rgb']) as image:
        if image.size != (depth.shape[1], depth.shape[0]):
            raise ValueError('RGB_DEPTH_SHAPE_MISMATCH')
    uv = np.asarray(pixels, dtype=float)
    if uv.size == 0:
        return dict(status='empty', pixels=[], points_body=[], points_odom=[], curve_body=[], curve_odom=[])
    if uv.shape != (5, 2) or not np.isfinite(uv).all():
        raise ValueError('WAYPOINT_REQUIRES_FIVE_FINITE_PIXEL_PAIRS')
    h, w = depth.shape
    if np.any(uv < 0) or np.any(uv[:, 0] >= w) or np.any(uv[:, 1] >= h):
        raise ValueError('WAYPOINT_OUTSIDE_SOURCE_RGB')
    # Containing original pixel (floor). No off-image clamping, synthetic ground plane,
    # or interpolation across invalid depth / object boundaries.
    ij = np.floor(uv).astype(int)
    z = depth[ij[:, 1], ij[:, 0]].astype(float)
    if not np.isfinite(z).all() or np.any(z <= .05) or np.any(z > 30):
        raise ValueError('INVALID_SAME_FRAME_DEPTH')
    K = np.asarray(source['K'], float)
    rays = np.c_[ij, np.ones(5)] @ np.linalg.inv(K).T
    body = transform_points(rays * z[:, None], source['T_body_camera'])
    world = transform_points(body, source['T_odom_body'])
    # This flat-floor warehouse protocol does not mistake a rack/sign surface
    # for a drivable ground target. Stairs require a different controller.
    if np.any(np.abs(world[:, 2]) > .15):
        raise ValueError('WAYPOINT_DEPTH_IS_NOT_OBSERVED_FLOOR')
    xy = body[:, :2]
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if np.any(ds < .03) or np.sum(ds) > 30:
        raise ValueError('DEGENERATE_OR_EXCESSIVE_PATH')
    # Fit only BETWEEN visible model anchors. The near-camera blind connector
    # is checked separately by the controller's current footprint/grid shield.
    s = np.r_[0., np.cumsum(ds)]
    curve = PchipInterpolator(s, xy, axis=0)(np.linspace(0, s[-1], max(2, int(s[-1]/.04)+1)))
    ground_body = transform_points(np.c_[transform_points(np.c_[curve, np.zeros(len(curve))], source['T_odom_body'])[:, :2],
                                               np.zeros(len(curve))], np.linalg.inv(source['T_odom_body']))
    camera = transform_points(ground_body, np.linalg.inv(source['T_body_camera']))
    projected = camera @ K.T
    projected = projected[:, :2] / projected[:, 2:3]
    inside = ((camera[:, 2] > .05) & (projected[:, 0] >= 0) & (projected[:, 0] < w)
              & (projected[:, 1] >= 0) & (projected[:, 1] < h))
    if not inside.all():
        raise ValueError('FITTED_CURVE_LEAVES_RGB')
    cells = np.floor(projected).astype(int)
    measured = depth[cells[:, 1], cells[:, 0]]
    if not np.isfinite(measured).all() or np.any(measured <= .05) or np.any(camera[:, 2] > measured + .15):
        raise ValueError('FITTED_CURVE_OCCLUDED_OR_DEPTH_INVALID')
    odom_curve = transform_points(ground_body, source['T_odom_body'])
    return dict(status='projected', pixels=uv.tolist(), sampled_pixels=ij.tolist(), optical_depth=z.tolist(),
                points_body=body.tolist(), points_odom=world.tolist(),
                curve_body=ground_body[:, :2].tolist(), curve_odom=odom_curve[:, :2].tolist(),
                curve_pixels=projected.tolist(), protocol=PROTOCOL,
                near_camera_connector='requires_current_observed_map_and_footprint_check',
                reference_decoder='not_run_engineering_PCHIP_curve')


@dataclass
class PeriodicLatestRequest:
    """One in flight + latest pending; periodic tick independent of old endpoint."""
    period_s: float = 2.0
    next_due: float = -float('inf')
    active: object = None
    pending: object = None
    replaced: int = 0

    def offer(self, now, source):
        if now < self.next_due:
            return False
        self.next_due = now + self.period_s
        if self.pending is not None:
            self.replaced += 1
        self.pending = source
        return True

    def take(self):
        if self.active is not None or self.pending is None:
            return None
        self.active, self.pending = self.pending, None
        return self.active

    def finish(self, source_id):
        if self.active is None or self.active['image_id'] != source_id:
            raise ValueError('OUT_OF_ORDER_OR_WRONG_SOURCE_RESPONSE')
        self.active = None
