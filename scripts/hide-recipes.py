from pathlib import Path
import re
import shutil

import yaml


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "source"


def main() -> None:
    toc_path = SOURCE / "_toctree.yml"
    toc = yaml.safe_load(toc_path.read_text())
    toc_path.write_text(
        yaml.safe_dump(
            [section for section in toc if section.get("title") != "Recipes"],
            sort_keys=False,
            allow_unicode=True,
        )
    )

    index_path = SOURCE / "index.md"
    index = re.sub(
        r'  <a\b[^>]*href="\./recipes/index"[^>]*>.*?</a>\n',
        "",
        index_path.read_text(),
        flags=re.DOTALL,
    )
    index_path.write_text(index.replace("lg:grid-cols-4", "lg:grid-cols-3"))

    dlcs_path = SOURCE / "get-started" / "dlcs.md"
    dlcs_path.write_text(
        dlcs_path.read_text().replace(
            " For tested deployments of popular models with the GPU image, "
            "see the [Recipes](../recipes/index).",
            "",
        )
    )
    shutil.rmtree(SOURCE / "recipes")


if __name__ == "__main__":
    main()
