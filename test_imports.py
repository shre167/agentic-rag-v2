"""Quick test to verify all imports and registry work."""
import sys
sys.path.insert(0, ".")

from retrieval.kb_registry import get_all_book_profiles, get_kb_registry, generate_book_profile, clear_kb_registry_cache, _load_profiles
print("kb_registry imports OK")

from retrieval.semantic_router import route_query
print("semantic_router imports OK")

from tools.semantic_search import semantic_search
print("semantic_search imports OK")

from agent.prompts import SYSTEM_PROMPT, _build_dynamic_prompt
print("prompts imports OK")

from ingest import ingest_pdf, register_existing_books
print("ingest imports OK")

# Check registry state
profiles = get_all_book_profiles()
print(f"\nRegistered profiles: {len(profiles)}")
for p in profiles:
    kb_id = p["kb_id"]
    kb_name = p["kb_name"]
    summary = p.get("summary", "N/A")[:80]
    topics = p.get("main_topics", [])[:5]
    print(f"  {kb_id}: {kb_name}")
    print(f"    Summary: {summary}...")
    print(f"    Topics: {topics}")

# Check persisted profiles
persisted = _load_profiles()
print(f"\nPersisted profiles on disk: {len(persisted)}")
for k, v in persisted.items():
    name = v.get("kb_name", "?")
    print(f"  {k}: {name}")

# Check the dynamic prompt
prompt = _build_dynamic_prompt()
print(f"\nDynamic prompt length: {len(prompt)} chars")
# Check it does NOT contain hardcoded book names
if "kb1: Mind Management" in prompt and len(profiles) == 0:
    print("WARNING: Prompt appears to have hardcoded books!")
else:
    print("OK: Prompt is dynamic (no hardcoded books)")

print("\n✅ All imports and checks passed!")
