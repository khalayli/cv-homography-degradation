
from src.matchers.orb import ORBMatcher
from src.matchers.xfeat import XFeatMatcher
from src.matchers.proposed import ProposedMatcher


def build_matcher(method_cfg):
    print("[build_matcher] Building matcher...")
    print(f"[build_matcher] method_cfg={method_cfg}")

    if method_cfg is None:
        raise ValueError("method_cfg cannot be None")

    method_name = method_cfg.get("name", None)

    if method_name is None:
        raise ValueError("method_cfg must contain a 'name' field")

    print(f"[build_matcher] method_name={method_name}")

    if method_name == "orb":
        print("[build_matcher] Creating ORBMatcher")
        return ORBMatcher(method_cfg)

    if method_name == "xfeat":
        print("[build_matcher] Creating XFeatMatcher")
        return XFeatMatcher(method_cfg)

    if method_name == "proposed":
        print("[build_matcher] Creating ProposedMatcher")
        return ProposedMatcher(method_cfg)

    raise ValueError(f"Unknown matcher name: {method_name}")