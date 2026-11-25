import hashlib
import json
import py_compile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositorySmokeTest(unittest.TestCase):
    def test_python_sources_compile(self):
        for path in sorted((ROOT / "code").rglob("*.py")):
            py_compile.compile(path, doraise=True)

    def test_notebooks_are_valid_json(self):
        for path in sorted((ROOT / "code" / "analysis").glob("*.ipynb")):
            with path.open(encoding="utf-8") as handle:
                notebook = json.load(handle)
            self.assertEqual(notebook.get("nbformat"), 4, path)
            source = "".join(
                "".join(cell.get("source", [])) for cell in notebook["cells"]
            )
            self.assertNotIn("/home/", source, path)
            self.assertNotIn("masks/mmr_K30_lam0.6.pth", source, path)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", source, path)
            self.assertNotIn("./snmf_analysis1", source, path)

    def test_artifact_manifest(self):
        manifest = ROOT / "artifacts" / "MANIFEST.sha256"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split(maxsplit=1)
            path = ROOT / relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, digest, relative)

    def test_expected_main_masks_exist(self):
        masks = ROOT / "artifacts" / "masks"
        self.assertTrue((masks / "llama2_topk_k30.pth").is_file())
        self.assertTrue((masks / "llama2_mmr_k30_lambda0.6.pth").is_file())
        self.assertTrue((masks / "llama3.1_8b_topk_k40.pth").is_file())


if __name__ == "__main__":
    unittest.main()
