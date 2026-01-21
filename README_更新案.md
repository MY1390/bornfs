# README.md 更新案

本ドキュメントは、現在のREADME.mdに追加・修正すべき内容を精査したものです。

---

## 追加すべきセクション

### 1. **Usage / 使用方法セクション**（詳細なコマンド例）

現在のREADMEには基本的な実行方法しか記載されていないため、詳細な使用方法を追加すべきです。

```markdown
## Usage

### Basic Usage

```sh
sbt "run -i data/test -o output/result.arff"
```

### Command Line Options

| オプション | 短縮形 | デフォルト値 | 説明 |
|-----------|--------|------------|------|
| `--input` | `-i` | (必須) | 入力ARFFファイルのパス（拡張子なし） |
| `--output` | `-o` | なし | 出力ARFFファイルのパス |
| `--threshold` | `-t` | 1.0 | 特徴選択の閾値（相互情報量の比率） |
| `--hop` | `-h` | 1 | 特徴再ソートの頻度 |
| `--sort` | `-s` | ratio | ソート基準（ratio, noise, relevance, difference, harmonic） |
| `--log` | `-l` | low | ログレベル（high, low, none） |
| `--tutorial` | `-T` | false | チュートリアルモード |
| `--verbose` | `-v` | true | 詳細表示モード |

### Examples

#### 1. 基本的な特徴選択
```sh
sbt "run -i data/test"
```

#### 2. 閾値を指定して実行
```sh
sbt "run -i data/test -t 0.8 -o output/selected_features.arff"
```

#### 3. チュートリアルモードで学習
```sh
sbt "run -T true -i data/test"
```

#### 4. ログを詳細出力
```sh
sbt "run -i data/test -l high"
```

#### 5. ソート基準を変更
```sh
sbt "run -i data/test -s relevance"
```
```

---

### 2. **CSV Support / CSV形式のサポート**

CSVReaderを実装したため、CSV形式のサポートについて記載すべきです。

```markdown
## CSV Format Support

BornFSは、ARFF形式に加えてCSV形式のデータもサポートしています。

### CSV to ARFF Conversion

CSVファイルをARFF形式に変換するには、提供されているPythonスクリプトを使用できます：

```sh
cd data
python3 arff_to_csv.py
```

### CSVReader in Scala

Scala内でCSVファイルを直接読み込むこともできます：

```scala
import scala.collection.mutable

val reader = CSVReader("data/test.csv")
val data = reader.sparse_instances
  .to(mutable.ArrayBuffer)
  .map { x =>
    (x._1.map(y => (reader.attr2index(y._1), y._2)), x._2)
  }
  .toSeq

val ds = Dataset(data, sort, tutorial, verbose)
val result = ds.select(threshold, hop)
```

詳細は[CSVReader使用例.md](CSVReader使用例.md)を参照してください。
```

---

### 3. **Testing / テストセクション**

テストスイートについての情報が欠けているため、追加すべきです。

```markdown
## Testing

BornFSには包括的なテストスイートが含まれています。

### Run All Tests

```sh
sbt test
```

### Run Specific Test

```sh
sbt "testOnly CSVReaderTest"
sbt "testOnly DataSetupTest"
```

### Test Coverage

プロジェクトには以下のテストが含まれています：

- **CSVReaderTest**: CSV読み込み機能のテスト
- **DataSetupTest**: ARFFデータの読み込みと特徴選択のテスト
- **CaseCompareSpec**: Caseクラスの比較ロジックのテスト
- **HopZeroSpec**: hop=0の場合の動作テスト
- **MultiDatasetSpec**: 多値特徴のテスト
- **ZeroMutualInfoSpec**: 相互情報量が0の場合の処理テスト

### Test Data

テストデータは`data/`ディレクトリに含まれています：
- `test.arff`: 小規模テストデータ（ARFF形式）
- `test.csv`: 小規模テストデータ（CSV形式）
- `multi.arff`: 多値特徴を含むテストデータ
- `dorothea.sparse.arff`: 大規模スパースデータ
```

---

### 4. **Project Structure / プロジェクト構造**

コードベースの構造を理解しやすくするため、追加すべきです。

```markdown
## Project Structure

```
bornfs/
├── src/
│   ├── main/
│   │   └── scala/
│   │       ├── bornfs.scala      # 主要アルゴリズム実装（Case, Dataset）
│   │       ├── Reader.scala      # ARFFReader, CSVReaderの実装
│   │       └── package.scala     # 型エイリアスと定数定義
│   └── test/
│       └── scala/
│           ├── CSVReaderTest.scala
│           ├── DataSetupTest.scala
│           ├── CaseCompareSpec.scala
│           ├── HopZeroSpec.scala
│           ├── MultiDatasetSpec.scala
│           └── ZeroMutualInfoSpec.scala
├── data/                         # テストデータとサンプルデータ
│   ├── test.arff
│   ├── test.csv
│   ├── multi.arff
│   ├── dorothea.sparse.arff
│   └── arff_to_csv.py           # ARFF→CSV変換スクリプト
├── compute_stats.py              # ログファイル解析用スクリプト
├── build.sbt                     # SBTビルド設定
├── README.md
├── CLAUDE.md                     # プロジェクト設定（日本語対応）
└── CSVReader使用例.md            # CSVReader使用方法
```
```

---

### 5. **Dependencies / 依存関係**

現在のREADMEには依存ライブラリの情報が不足しているため、追加すべきです。

```markdown
## Dependencies

このプロジェクトは以下のライブラリに依存しています：

### Scala Libraries
- **Scala**: 3.3.1
- **Weka**: 3.7.12 - データ処理とARFF形式のサポート
- **scopt**: 4.1.0 - コマンドライン引数解析
- **scala-parallel-collections**: 1.0.4 - 並列コレクション処理

### Testing Libraries
- **ScalaTest**: 3.2.19 - テストフレームワーク
- **Scalactic**: 3.2.19 - テスト用のマッチャーとアサーション

### Build Tool
- **SBT**: 1.x - ビルドツール

すべての依存関係は`build.sbt`に定義されており、SBTが自動的に解決します。
```

---

### 6. **Output Files / 出力ファイル**

BornFSが生成するファイルについての説明が欠けています。

```markdown
## Output Files

BornFSは以下のファイルを生成します：

### 1. Selected Feature ARFF File（オプション）

`-o`オプションで指定した場合、選択された特徴のみを含むARFFファイルを出力します。

```sh
sbt "run -i data/test -o output/selected.arff"
```

### 2. Log File

特徴選択の詳細なログファイルが自動的に生成されます：

**ファイル名形式**: `{input}-{sort}-{threshold}-{hop}-bornfs.log`

**例**: `data/test-ratio-1.0-1-bornfs.log`

### Log File Contents

ログファイルには以下の情報が含まれます：

- **Parameters**: 実行時のパラメータ
- **Run-time**: 各処理ステップの実行時間
- **Selected features**: 選択された特徴のリスト
- **Statistics**: エントロピー、相互情報量などの統計情報
- **Events**: 処理イベントのタイムスタンプ

### Log Levels

- **high**: 詳細ログ（各特徴のエントロピー値を含む）
- **low** (デフォルト): 標準ログ（選択結果と基本統計）
- **none**: ログファイルを生成しない
```

---

### 7. **Algorithm Details / アルゴリズムの詳細**

アルゴリズムの特徴をより詳しく説明すべきです。

```markdown
## Algorithm Details

### Sorting Measures

BornFSは、特徴を評価するための5つの異なる尺度を提供します：

1. **ratio** (デフォルト): 関連性とノイズの比率
   - 最もバランスの取れた尺度
   - 推奨される設定

2. **noise**: ノイズゲインの負値
   - ノイズが少ない特徴を優先

3. **relevance**: 関連性ゲイン
   - クラスラベルとの関連性が高い特徴を優先

4. **difference**: 関連性 - ノイズ
   - 関連性とノイズの差分

5. **harmonic**: 調和平均
   - 関連性とノイズの調和平均

### Threshold Parameter

`-t`オプションで設定する閾値は、選択する特徴の相互情報量の比率を制御します：

- **1.0** (デフォルト): 最も厳格（少数の重要な特徴のみ選択）
- **0.8**: 中程度の厳格さ
- **0.5**: より多くの特徴を選択
- **0.0**: すべての特徴を選択

### Hop Parameter

`-h`オプションは、特徴の再ソート頻度を制御します：

- **0**: ソートを最初に1回だけ実行
- **1** (デフォルト): 毎イテレーションでソート
- **n**: n回のイテレーションごとにソート

値が大きいほど計算が高速になりますが、精度が低下する可能性があります。
```

---

### 8. **Performance Tips / パフォーマンスのヒント**

大規模データセットでの使用に関するアドバイスを追加すべきです。

```markdown
## Performance Tips

### Large Datasets

大規模データセット（数千〜数万の特徴）を処理する場合：

1. **hopパラメータを調整**
   ```sh
   sbt "run -i data/large_dataset -h 10"
   ```

2. **verbose モードをオフに**
   ```sh
   sbt "run -i data/large_dataset -v false"
   ```

3. **ログレベルを下げる**
   ```sh
   sbt "run -i data/large_dataset -l none"
   ```

### Memory Settings

大規模データセットの場合、JVMのメモリ設定を増やすことを推奨します：

```sh
export SBT_OPTS="-Xmx4G"
sbt "run -i data/large_dataset"
```

### Sparse Data

スパースデータ（多くの0値を含むデータ）の場合、BornFSは自動的にメモリ効率の良いスパース表現を使用します。
```

---

### 9. **Additional Tools / 追加ツール**

compute_stats.pyについての説明を追加すべきです。

```markdown
## Additional Tools

### Log File Analysis

`compute_stats.py`スクリプトを使用して、BornFSのログファイルから統計情報を抽出できます：

```sh
python3 compute_stats.py --help
```

このスクリプトは以下の情報を計算します：
- 選択された特徴の統計
- 特徴選択の再現性
- 複数実行間の一貫性

### ARFF to CSV Conversion

ARFFファイルをCSV形式に変換するには：

```sh
cd data
python3 arff_to_csv.py
```

変換されたCSVファイルは、元のARFFファイルと同じディレクトリに生成されます。
```

---

### 10. **Installation セクションの修正**

現在のインストール手順が不完全なため、修正が必要です。

**現在の記述**:
```markdown
```sh
git clone https://github.com/yourusername/bornfs.git
cd bornfs
sbt run -i xxx
```
```

**推奨される修正**:
```markdown
### Clone the Repository

```sh
git clone https://github.com/yourusername/bornfs.git
cd bornfs
```

### Compile the Project

```sh
sbt compile
```

### Run Tests

```sh
sbt test
```

### Run BornFS

```sh
sbt "run -i data/test"
```
```

---

### 11. **Troubleshooting / トラブルシューティング**

よくある問題と解決方法を追加すべきです。

```markdown
## Troubleshooting

### Common Issues

#### 1. "Missing option --input" エラー

**原因**: 入力ファイルが指定されていません。

**解決方法**:
```sh
sbt "run -i data/test"
```

#### 2. "File not found" エラー

**原因**: 入力ファイルのパスが正しくありません。

**解決方法**:
- ファイルパスを確認してください
- ARFF形式の場合、拡張子`.arff`を省略してください
  ```sh
  # 正しい
  sbt "run -i data/test"

  # 誤り
  sbt "run -i data/test.arff"
  ```

#### 3. OutOfMemoryError

**原因**: JVMのヒープサイズが不足しています。

**解決方法**:
```sh
export SBT_OPTS="-Xmx8G"
sbt "run -i data/large_dataset"
```

#### 4. コンパイルエラー

**原因**: 依存関係が正しく解決されていない可能性があります。

**解決方法**:
```sh
sbt clean
sbt compile
```
```

---

## セクションの推奨順序

更新後のREADME.mdは以下の順序で構成することを推奨します：

1. タイトルとバッジ
2. Overview（既存）
3. Features（既存）
4. Prerequisites（既存）
5. **Dependencies**（新規）
6. Data Requirements（既存）
7. Installation（修正版）
8. **Usage**（新規・詳細版）
9. **CSV Format Support**（新規）
10. **Algorithm Details**（新規）
11. **Output Files**（新規）
12. Tutorial（既存）
13. **Testing**（新規）
14. **Performance Tips**（新規）
15. **Additional Tools**（新規）
16. **Project Structure**（新規）
17. **Troubleshooting**（新規）
18. Reference（既存）
19. Questions and Issues（既存）

---

## その他の推奨事項

### 1. リポジトリURLの修正

現在のREADMEには`yourusername`というプレースホルダーが残っています：
```markdown
git clone https://github.com/yourusername/bornfs.git
```

実際のリポジトリURLに置き換える必要があります。

### 2. バッジの追加検討

以下のバッジを追加することを検討してください：
- ![License](https://img.shields.io/badge/License-...-blue.svg)
- ![Build Status](...)
- ![Test Coverage](...)

### 3. 日本語版READMEの作成

CLAUDE.mdに記載されているように、日本語でのコミュニケーションが前提の場合、`README_ja.md`を作成することを推奨します。

### 4. Examples ディレクトリの作成

実際の使用例を示すスクリプトやノートブックを`examples/`ディレクトリに配置することを推奨します。

---

## まとめ

本更新案では、以下の点を改善しています：

1. ✅ **使用方法の詳細化**: コマンドラインオプションの完全な説明
2. ✅ **CSV対応の追加**: 新しく実装されたCSVReader機能の説明
3. ✅ **テスト情報の追加**: テストスイートの実行方法と内容
4. ✅ **プロジェクト構造の明示**: コードベースの理解を容易に
5. ✅ **依存関係の明確化**: 使用しているライブラリの一覧
6. ✅ **出力ファイルの説明**: BornFSが生成するファイルの詳細
7. ✅ **アルゴリズムの詳細**: パラメータと尺度の説明
8. ✅ **パフォーマンスガイド**: 大規模データセットでの最適化
9. ✅ **トラブルシューティング**: よくある問題と解決方法
10. ✅ **追加ツールの説明**: compute_stats.pyなどのユーティリティ

これらの追加により、README.mdはより包括的で、新しいユーザーにとって理解しやすいドキュメントになります。
