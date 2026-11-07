"""Numerical checks and offline end-to-end tests; run: python -m unittest discover -s tests -v."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataio import read_rows
from metrics import evaluate
from model import LABELS, Softmax, Tfidf, load, loss_gradient, route, save
from text import is_billing, is_urgent, normalize


class NumericalTests(unittest.TestCase):
    def test_idf_tf_normalization_and_unseen_features(self):
        vectorizer = Tfidf().fit(["a a", "b"])
        self.assertAlmostEqual(vectorizer.idf[vectorizer.vocabulary["w:a"]], np.log(3 / 2) + 1)
        x = vectorizer.transform(["a a", "b", "unseen"])
        np.testing.assert_allclose(np.linalg.norm(x[:2], axis=1), 1)
        np.testing.assert_array_equal(x[2], 0)
        before = dict(vectorizer.vocabulary)
        vectorizer.transform(["validationonlytoken"])
        self.assertEqual(before, vectorizer.vocabulary)

    def test_softmax_gradient_against_finite_differences(self):
        rng = np.random.default_rng(7)
        x, y = rng.normal(size=(4, 3)), np.array([0, 2, 4, 1])
        weights, bias = rng.normal(size=(3, 5)), rng.normal(size=5)
        _, dw, db = loss_gradient(x, y, weights, bias, 0.03)
        for parameter, gradient in ((weights, dw), (bias, db)):
            for index in np.ndindex(parameter.shape):
                original, epsilon = parameter[index], 1e-6
                parameter[index] = original + epsilon
                plus = loss_gradient(x, y, weights, bias, 0.03)[0]
                parameter[index] = original - epsilon
                minus = loss_gradient(x, y, weights, bias, 0.03)[0]
                parameter[index] = original
                self.assertAlmostEqual((plus - minus) / (2 * epsilon), gradient[index], places=6)

    def test_large_logits_remain_finite(self):
        loss, dw, db = loss_gradient(np.array([[1.0, -1.0]]), np.array([2]),
                                     np.arange(10).reshape(2, 5) * 1000.0,
                                     np.array([10000.0, -10000.0, 0, 0, 0]), 0)
        self.assertTrue(np.isfinite(loss) and np.isfinite(dw).all() and np.isfinite(db).all())

    def test_learning_reduces_loss_and_round_trip_preserves_predictions(self):
        texts = ["gaz bor", "suv bor", "axlat bor", "tok bor", "tolov bor"]
        actual = np.array(["gas", "water", "waste", "electricity", "billing"])
        vectorizer = Tfidf().fit(texts)
        model = Softmax().fit(vectorizer.transform(texts), np.searchsorted(LABELS, actual), epochs=100)
        self.assertLess(model.losses[-1], model.losses[0])
        np.testing.assert_array_equal(route(model, vectorizer, texts), actual)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "model.npz"
            save(path, vectorizer, model)
            loaded_vectorizer, loaded_model = load(path)
            np.testing.assert_array_equal(route(loaded_model, loaded_vectorizer, texts), actual)

    def test_metrics_with_errors_and_absent_classes(self):
        report = evaluate(["billing", "billing", "gas", "gas", "water"],
                          ["billing", "gas", "gas", "water", "water"])
        self.assertAlmostEqual(report["accuracy"], 0.6)
        self.assertAlmostEqual(report["macro_f1"], 11 / 30)
        self.assertEqual(report["confusion_matrix"][0], [1, 0, 1, 0, 0])
        self.assertEqual(report["per_class"]["electricity"]["f1"], 0)


class TextTests(unittest.TestCase):
    def test_scripts_and_apostrophes(self):
        self.assertEqual(normalize("Газ ҳиди келяпти"), normalize("Gaz hidi kelyapti"))
        self.assertEqual(normalize("Тўлов қилдим"), normalize("To‘lov qildim"))
        for apostrophe in ("'", "‘", "’", "ʻ", "ʼ", "`", "´"):
            self.assertEqual(normalize("to" + apostrophe + "lov"), "tolov")
        self.assertEqual(normalize("Нет воды"), "yoq suv")
        self.assertEqual(normalize("Счётчик газа"), "hisoblagich gaza")
        self.assertEqual(normalize("Ёмғир"), "yomgir")

    def test_billing_precedes_service_but_not_meter(self):
        for text in ("Gaz uchun hisob katta", "Elektr uchun qarzim bor", "Click orqali to‘lay olmadim",
                     "Suv uchun to'lov", "Счёт за газ", "Задолженность за воду"):
            self.assertTrue(is_billing(text), text)
        for text in ("Gaz hisoblagichini tekshiring", "Suv hisoblagich ko'rsatkichi",
                     "Konteyner to'lib ketgan", "Счётчик газа сломан"):
            self.assertFalse(is_billing(text), text)

    def test_urgency_positive_negative_and_mixed_clauses(self):
        for text in ("Газ ҳиди келяпти", "Gaz quvuri yorilgan", "Sim uzilib yotibdi",
                     "Ustundagi sim osilib qolgan", "Yerto'lani suv bosgan", "Tok urdi",
                     "Transformator yonida uchqun chiqdi", "Запах газа", "Оборван провод", "Прорвало трубу",
                     "Gaz hidi yo'q, lekin sim uzilgan"):
            self.assertEqual(is_urgent(text), 1, text)
        for text in ("Gaz bosimi past", "Suv hisoblagichini tekshiring", "Axlatdan hid chiqyapti",
                     "Gaz hidi yo'q", "Газ ҳиди йўқ", "Нет запаха газа", "Gaz uchun qarzim bor"):
            self.assertEqual(is_urgent(text), 0, text)


class CLITests(unittest.TestCase):
    def run_cli(self, folder, source):
        return subprocess.run([sys.executable, str(ROOT / "predict.py"), str(source), "result.csv"],
                              cwd=folder, capture_output=True, text=True,
                              env=dict(os.environ, PYTHONUTF8="1"), timeout=30)

    def test_offline_cli_from_another_directory_with_utf8_bom_and_ids(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            source = Path(folder) / "input.csv"
            with source.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "text"])
                writer.writerows([("0001", "Газ ҳиди келяпти"), ("0002", "Нет воды"),
                                  ("0003", "Gaz uchun hisob katta, xato bo'lsa kerak")])
            process = self.run_cli(folder, source)
            self.assertEqual(process.returncode, 0, process.stderr)
            with (Path(folder) / "result.csv").open(encoding="utf-8", newline="") as handle:
                result = list(csv.DictReader(handle))
            self.assertEqual(result, [
                {"id": "0001", "label": "gas", "is_urgent": "1"},
                {"id": "0002", "label": "water", "is_urgent": "0"},
                {"id": "0003", "label": "billing", "is_urgent": "0"}])

    def test_empty_input_and_empty_text(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            source = Path(folder) / "input.csv"
            source.write_text("id,text\n", encoding="utf-8")
            self.assertEqual(self.run_cli(folder, source).returncode, 0)
            self.assertEqual((Path(folder) / "result.csv").read_text().strip(), "id,label,is_urgent")
            source.write_text('id,text\nempty,""\n', encoding="utf-8")
            self.assertEqual(self.run_cli(folder, source).returncode, 0)
            with (Path(folder) / "result.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_bad_header_and_incomplete_row_fail_without_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            source = Path(folder) / "input.csv"
            for content in ("id,message\n1,hello\n", "id,text\n1\n"):
                source.write_text(content, encoding="utf-8")
                process = self.run_cli(folder, source)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("Error:", process.stderr)
                self.assertFalse((Path(folder) / "result.csv").exists())


if __name__ == "__main__":
    unittest.main()
