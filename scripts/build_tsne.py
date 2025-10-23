import json
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from pathlib import Path
from ariautils.midi import MidiDict

from aria.embedding import get_global_embedding_from_midi, _get_chunks
from aria.model import TransformerEMB, ModelConfig
from aria.config import load_model_config
from ariautils.tokenizer import AbsTokenizer
from safetensors.torch import load_file

COMPOSITIONS_JSON_PATH = (
    "/home/loubb/work/aria/blog/figure/aria_embeddings.json"
)
METADATA_JSON_PATH = (
    "/mnt/ssd1/aria-midi/final/v1/aria-midi-v1-int/metadata.json"
)
MIDI_DIR = "/mnt/ssd1/aria-midi/final/v1/aria-midi-v1-int/data"
CHECKPOINT_PATH = "/mnt/ssd1/aria/v2/emb-t0.1-s2048-aug-ds-large/checkpoints/epoch25_step0/model.safetensors"

MIDI_CHUNK_SAVE_DIR = Path("./mid")
FINAL_JSON_SAVE_PATH = "data.json"
TSNE_PLOT_SAVE_PATH = "tsne.png"

COMPOSER_WHITELIST = [
    "scriabin",
    "chopin",
    "schubert",
    "bach",
    "schumann",
    "haydn",
    "beethoven",
    "mozart",
    "satie",
    "brahms",
    "tchaikovsky",
    "liszt",
    "debussy",
    "rachmaninoff",
    "ravel",
    "handel",
]


tokenizer = AbsTokenizer()


def _process_chunk(chunk: MidiDict) -> MidiDict:

    chunk_p = tokenizer.detokenize(
        tokenizer.tokenize(chunk.remove_redundant_pedals())
    )

    return chunk_p


def load_embedding_model():
    model_config = ModelConfig(**load_model_config(name="medium-emb"))
    model_config.set_vocab_size(AbsTokenizer().vocab_size)
    model = TransformerEMB(model_config)

    # Load state dict
    state_dict = load_file(filename=CHECKPOINT_PATH)
    model.load_state_dict(state_dict=state_dict, strict=True)
    model.cuda().eval()

    return model


def get_midi_path(index: str, segment: str, midi_dir=MIDI_DIR) -> Path:
    midi_dir = Path(midi_dir)
    filename = f"{int(index):06d}_{segment}.mid"
    for entry in midi_dir.iterdir():
        if not entry.is_dir():
            continue
        candidate = entry / filename
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"{filename} not found under {midi_dir}")


def generate_tsne(
    embeddings: np.ndarray, compositions: list, plot_path: str = "tsne.png"
):
    """
    Runs t-SNE on the provided embeddings and saves a verification plot
    colored by composer.
    """
    print(f"\nRunning t-SNE on {len(embeddings)} embeddings...")
    tsne = TSNE(
        n_components=2,
        perplexity=50,
        init="pca",
        learning_rate="auto",
        max_iter=500,
    )
    tsne_results = tsne.fit_transform(embeddings)
    print("t-SNE complete.")

    # --- Modified Section ---

    # Extract composer labels from the compositions list
    labels = [comp.get("composer", "Unknown") for comp in compositions]
    unique_labels = sorted(list(set(labels)))

    # Create a color map for the unique composers
    # Using 'tab10' colormap which is good for 10 categories
    colors = plt.cm.get_cmap("tab10", len(unique_labels))
    label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}

    # Save verification plot
    print(f"Saving verification plot to {plot_path}...")
    # Make figure slightly wider to accommodate the legend
    plt.figure(figsize=(13, 10))

    # Plot each composer group separately to build the legend
    for label in unique_labels:
        # Find the indices for all points belonging to this composer
        indices = [i for i, l in enumerate(labels) if l == label]

        # Select the t-SNE coordinates for these points
        x = tsne_results[indices, 0]
        y = tsne_results[indices, 1]

        plt.scatter(
            x,
            y,
            alpha=0.7,
            s=15,
            color=label_to_color[label],
            label=label.title(),  # Add capitalized label for the legend
        )

    plt.title("t-SNE of Piano Compositions (Colored by Composer)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)

    # Add the legend outside the plot
    plt.legend(title="Composer", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()  # Adjust plot to make room for the legend
    plt.savefig(plot_path)

    # --- End Modified Section ---

    print("Plot saved.")

    return tsne_results


def find_best_midi_matches(
    compositions: list, metadata: dict, model: TransformerEMB
):
    print(
        f"\nFinding best MIDI matches for {len(compositions)} compositions..."
    )
    best_segments = {}  # To store results

    for idx, composition in enumerate(compositions):
        comp_composer = composition.get("composer")
        comp_opus = composition.get("opus")
        comp_piece_num = composition.get("piece_number")

        comp_emb_tensor = torch.tensor(
            composition["emb"],
            dtype=torch.bfloat16,
            device="cuda",
        )

        comp_key = (comp_composer, comp_opus, comp_piece_num)
        print(f"\n[{idx+1}/{len(compositions)}] Processing: {comp_key}")

        # 1. Find all matching segments and their scores
        potential_segments = []
        for index, data in metadata.items():
            meta = data.get("metadata", {})

            # Check for a metadata match
            is_match = (
                meta.get("composer") == comp_composer
                and meta.get("opus") == comp_opus
                and meta.get("piece_number") == comp_piece_num
            )

            if is_match:
                for segment, score in data.get("audio_scores", {}).items():
                    potential_segments.append((score, index, segment))

        if not potential_segments:
            print(
                f"  Warning: No matching audio segments found for {comp_key}."
            )
            continue

        potential_segments.sort(key=lambda x: x[0], reverse=True)
        top_5_segments = potential_segments[:10]
        print(
            f"  Found {len(potential_segments)} segments. Analyzing top {len(top_5_segments)}."
        )

        best_similarity = -float("inf")
        best_path = None

        for score, index, segment in top_5_segments:
            midi_path = get_midi_path(index, segment)

            if not midi_path.exists():
                print(f"    Skipping: {midi_path} not found.")
                continue

            midi_emb = get_global_embedding_from_midi(
                model=model,
                midi_path=str(midi_path),
                device="cuda",
            )

            # (Shape prints removed for brevity)
            similarity = torch.nn.functional.cosine_similarity(
                comp_emb_tensor, midi_emb, dim=0
            )

            print(
                f"    - {midi_path.name}: (Audio Score: {score:.4f}, Embedding Sim: {similarity.item():.4f})"
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_path = midi_path

        if best_path:
            if best_similarity.item() < 0.8:
                print(
                    f"  Best File: {best_path} (Global sim: {best_similarity.item():.4f}). Skipping..."
                )
                continue
            else:
                print(
                    f"  Best File: {best_path} (Global sim: {best_similarity.item():.4f}). Finding best chunk..."
                )

            midi_dict = MidiDict.from_midi(str(best_path))
            notes_per_chunk = 300  # As per your function example
            chunks = _get_chunks(midi_dict, notes_per_chunk=notes_per_chunk)
            chunks = chunks[:-1]

            if not chunks:
                print(f"  Warning: No chunks found for {best_path}.")
                continue

            best_chunk_similarity = -float("inf")
            best_chunk_data = None

            for _idx, chunk_midi_dict in enumerate(chunks):
                if not chunk_midi_dict.note_msgs:
                    continue

                # Get embedding for the chunk
                chunk_emb = get_global_embedding_from_midi(
                    model=model,
                    midi_dict=chunk_midi_dict,
                    device="cuda",
                )

                chunk_sim = torch.nn.functional.cosine_similarity(
                    comp_emb_tensor, chunk_emb, dim=0
                )

                if chunk_sim > best_chunk_similarity:
                    best_chunk_similarity = chunk_sim

                    best_chunk_data = {
                        "original_file_path": str(best_path),
                        "chunk_idx": _idx,
                        "similarity": chunk_sim.item(),
                    }

            if best_chunk_data:
                comp_key_str = "_".join(
                    str(k).replace(" ", "_") for k in comp_key
                )
                chunk_filename = f"comp_{comp_key_str}__chunk_{best_chunk_data['chunk_idx']}.mid"
                chunk_save_path = MIDI_CHUNK_SAVE_DIR / chunk_filename
                best_chunk_dict_to_save = chunks[best_chunk_data["chunk_idx"]]
                best_chunk_dict_to_save = _process_chunk(
                    best_chunk_dict_to_save
                )
                best_chunk_dict_to_save.to_midi().save(str(chunk_save_path))
                best_chunk_data["chunk_file_path"] = str(chunk_save_path)

                best_segments[comp_key] = best_chunk_data
                print(
                    f"  Best Chunk Found: {best_chunk_data['chunk_idx']}, "
                    f"Sim: {best_chunk_data['similarity']:.4f}, "
                    f"Saved to: {chunk_save_path}"  # Modified print
                )
            else:
                print(
                    f"  Warning: No valid chunks with notes found for {best_path}."
                )

        else:
            print(
                f"  Warning: No valid MIDI files found for top 5 segments of {comp_key}."
            )

    return best_segments


def create_final_json(
    compositions: list,
    tsne_results: np.ndarray,
    best_segments: dict,
    save_path: str,
):
    print(f"\nCreating final JSON file at {save_path}...")
    final_data = []

    for i, composition in enumerate(compositions):
        tsne_coord = tsne_results[i]

        # Get composition key
        comp_key = (
            composition.get("composer"),
            composition.get("opus"),
            composition.get("piece_number"),
        )

        # Find the matching segment data
        segment_data = best_segments.get(comp_key)

        if segment_data is None:
            print(
                f"  Warning: No segment data found for {comp_key}. Skipping point."
            )
            continue

        # --- Prepare data for JSON ---

        # Get the MIDI chunk path and create the future MP3 path
        chunk_path = Path(segment_data["chunk_file_path"])
        mp3_filename = chunk_path.with_suffix(".mp3").name
        # This path is relative to the web page (e.g., in an 'audio' folder)
        mp3_path_for_web = f"audio/{mp3_filename}"

        # Create a display name for the piece
        piece_name = f"{composition.get('composer', 'Unknown').title()}"
        if composition.get("opus"):
            piece_name += f", Op. {composition.get('opus')}"
        if composition.get("piece_number"):
            piece_name += f", No. {composition.get('piece_number')}"

        data_point = {
            "x": float(tsne_coord[0]),  # t-SNE x
            "y": float(tsne_coord[1]),  # t-SNE y
            "cluster": composition.get("composer", "Unknown"),  # For coloring
            "piece": piece_name,  # For hover text
            "audioFile": mp3_path_for_web,  # Path to the audio file to play
        }

        final_data.append(data_point)

    # Save the final list of objects as a JSON file
    with open(save_path, "w") as f:
        json.dump(final_data, f, indent=2)

    print(f"Successfully saved {len(final_data)} data points to {save_path}.")


if __name__ == "__main__":
    MIDI_CHUNK_SAVE_DIR.mkdir(exist_ok=True)

    model = load_embedding_model()

    print(f"Loading data from {COMPOSITIONS_JSON_PATH}...")
    with open(COMPOSITIONS_JSON_PATH, "r") as f:
        all_compositions_data = json.load(f)

    print(
        f"Filtering {len(all_compositions_data)} compositions by whitelist..."
    )
    filtered_compositions = [
        comp
        for comp in all_compositions_data
        if comp.get("composer") in COMPOSER_WHITELIST
    ]
    print(f"Found {len(filtered_compositions)} matching compositions.")

    if not filtered_compositions:
        print("No compositions found matching the whitelist. Exiting.")
        exit()

    embeddings = np.array(
        [comp["emb"] for comp in filtered_compositions]
    ).astype(
        np.float32
    )  # TSNE prefers float32

    tsne_results = generate_tsne(
        embeddings,
        compositions=filtered_compositions,  # <-- Pass the compositions
        plot_path=TSNE_PLOT_SAVE_PATH,
    )

    print(f"Loading metadata from {METADATA_JSON_PATH}...")
    with open(METADATA_JSON_PATH, "r") as f:
        metadata_data = json.load(f)

    best_segments = find_best_midi_matches(
        compositions=filtered_compositions, metadata=metadata_data, model=model
    )

    create_final_json(
        compositions=filtered_compositions,
        tsne_results=tsne_results,
        best_segments=best_segments,
        save_path=FINAL_JSON_SAVE_PATH,
    )

    print(f"\n--- Complete ---")
    print(f"Final data saved to {FINAL_JSON_SAVE_PATH}")
    print(f"Verification plot saved to {TSNE_PLOT_SAVE_PATH}")
    print(f"MIDI chunks saved to {MIDI_CHUNK_SAVE_DIR}/")
