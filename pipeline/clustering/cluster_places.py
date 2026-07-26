"""Phase 7 — unsupervised clustering into "vibe collections".

KMeans on the place embeddings from Phase 6 (which already encode category,
neighborhood, and any linked review text - see build_place_text in
embed_places.py). K is picked via silhouette score over a small range rather
than a fixed guess. One cheap Groq call per cluster (not per place) names it
from its nearest-centroid exemplars, keeping LLM cost trivial regardless of
dataset size.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm" / "prompts"))
from groq_client import GroqError, call_groq  # noqa: E402
from cluster_naming_few_shot import CLUSTER_NAMING_PROMPT  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cluster_places")

K_CANDIDATES = [8, 10, 12, 15, 18]
EXEMPLARS_PER_CLUSTER = 5


def load_places(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT place_id, name, category, neighborhood, embedding
            FROM places WHERE embedding IS NOT NULL;
            """
        )
        columns = [d.name for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def pick_best_k(X: np.ndarray) -> tuple[int, KMeans]:
    best_k, best_score, best_model = None, -1, None
    for k in K_CANDIDATES:
        model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        score = silhouette_score(X, model.labels_)
        log.info(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_model = k, score, model
    log.info(f"Chose k={best_k} (silhouette={best_score:.4f})")
    return best_k, best_model


def name_cluster(exemplars: list[dict]) -> dict:
    exemplar_str = "\n".join(
        f"- {p['name']} ({p['category']}, {p['neighborhood'] or 'unknown area'})" for p in exemplars
    )
    prompt = CLUSTER_NAMING_PROMPT.format(exemplar_places=exemplar_str)
    try:
        return call_groq(prompt)
    except GroqError as e:
        log.warning(f"Groq naming failed, using a generic fallback label: {e}")
        categories = {p["category"] for p in exemplars}
        return {"label": f"{'/'.join(categories)} cluster", "description": "Auto-generated fallback label."}


def main():
    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        register_vector(conn)
        places = load_places(conn)
        log.info(f"Loaded {len(places)} places with embeddings")

        X = np.array([p["embedding"].to_numpy() for p in places], dtype=np.float32)
        k, model = pick_best_k(X)

        with conn.cursor() as cur:
            cur.execute("DELETE FROM place_clusters;")
            cur.execute("DELETE FROM clusters;")

            for cluster_idx in range(k):
                member_indices = [i for i, label in enumerate(model.labels_) if label == cluster_idx]
                centroid = model.cluster_centers_[cluster_idx]
                distances = [
                    (i, float(np.linalg.norm(X[i] - centroid))) for i in member_indices
                ]
                distances.sort(key=lambda x: x[1])
                exemplars = [places[i] for i, _ in distances[:EXEMPLARS_PER_CLUSTER]]

                naming = name_cluster(exemplars)
                log.info(f"Cluster {cluster_idx} ({len(member_indices)} places): {naming['label']}")

                cur.execute(
                    """
                    INSERT INTO clusters (label, description, model_params)
                    VALUES (%s, %s, %s) RETURNING cluster_id;
                    """,
                    (naming["label"], naming["description"], psycopg.types.json.Json({"k": k, "algorithm": "kmeans"})),
                )
                cluster_id = cur.fetchone()[0]

                for i, dist in distances:
                    cur.execute(
                        """
                        INSERT INTO place_clusters (place_id, cluster_id, distance_to_centroid)
                        VALUES (%s, %s, %s);
                        """,
                        (places[i]["place_id"], cluster_id, dist),
                    )
        conn.commit()
    log.info(f"Done. {k} clusters created and named.")


if __name__ == "__main__":
    main()
