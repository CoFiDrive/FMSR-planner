# Third-party sources

`vendor/garage/` is a local snapshot of CARLA Garage, retaining its original
headers. Its MIT license is included as `vendor/GARAGE_LICENSE`. The snapshot
was taken from this workspace's `3rd_party/carla_garage/team_code/` on
2026-10-05. Only the snapshot's image-pretraining switch and model import in
the agent were adapted for isolated operation.

`upstream/bridgedrive/` contains the published BridgeDrive LEAD adaptation
files at commit `85aa089a321a7b7cbc42aa58b3be508eb223d9e2`. Source URLs and
SHA256 values appear in `upstream/sources.json`. Author attribution and
upstream documentation are retained. `bridgepdm/vendor_ddbm/` adapts only
module imports and the BEV feature-channel parameter in those files; the
PDM-specific schedule, supervision and spatial attention live in separate
adapter modules. The upstream LEAD adaptation does not include a root
license file; no additional license grant is asserted here.

The workspace's existing TransFuser, LEAD, Bench2Drive, CARLA and other
dependency notices continue to apply to their respective components.
