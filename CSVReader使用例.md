# CSVReader 使用例

## 概要

`CSVReader`は、CSV形式のデータファイルを読み込み、`ARFFReader`と同様のインターフェースでスパース表現に変換するクラスです。

## 基本的な使い方

```scala
// CSVファイルを読み込む
val reader = CSVReader("data/test.csv")

// データセットの情報を取得
println(s"インスタンス数: ${reader.numInstances}")
println(s"属性数: ${reader.numAttrs}")

// 属性名からインデックスを取得
val index = reader.attr2index(Symbol("a"))

// インデックスから属性名を取得
val attrName = reader.index2attr(0)

// スパース表現のデータを取得
val sparseData = reader.sparse_instances
```

## データ構造

### CSVファイルの形式

```csv
a,b,c,d,e,f,class
0,1,0,0,1,0,0
1,1,0,0,1,0,1
...
```

- 1行目: ヘッダー（属性名）
- 2行目以降: データ行
- 最後の列: クラスラベル

### スパース表現

CSVReaderは、0以外の値を持つ属性のみを保持するスパース表現に変換します。

```scala
// スパース表現の例:
// (ArrayBuffer[(Symbol, Int)], Int)
//
// ArrayBuffer: 非ゼロ属性のリスト (属性名, 値)
// Int: クラスラベル

val sparse = reader.sparse_instances.head
// sparse._1: ArrayBuffer((Symbol("b"), 1), (Symbol("e"), 1))
// sparse._2: 0
```

## BornFSとの統合

CSVReaderで読み込んだデータは、ARFFReaderと同じようにBornFSの特徴選択に使用できます。

```scala
val reader = CSVReader("data/test.csv")

// スパース表現をBornFS内部形式に変換
val data = reader.sparse_instances
  .to(scala.collection.mutable.ArrayBuffer)
  .map { x =>
    (
      x._1.map(y => (reader.attr2index(y._1), y._2)),
      x._2
    )
  }
  .toSeq

// Datasetオブジェクトを作成
val ds = Dataset(data, sort, tutorial, verbose)

// 特徴選択を実行
val result = ds.select(threshold, hop)
val selected_attrs = result.map(i => reader.index2attr(i)).toList

// 選択された特徴のみを含むCSVファイルを保存
reader.saveCsvFile(selected_attrs, "data/test_selected.csv")
```

## ARFFReaderとの比較

| 機能 | ARFFReader | CSVReader |
|------|-----------|-----------|
| データ形式 | ARFF (Weka形式) | CSV |
| 依存ライブラリ | Weka | Scala標準ライブラリ |
| スパース表現対応 | ○ | ○ |
| 属性マッピング | ○ | ○ |
| ファイル保存 | saveArffFile() | saveCsvFile() |

## 主なメソッド

### `CSVReader(filename: String)`

CSVファイルを読み込んでCSVReaderオブジェクトを作成します。

### `attr2index: HashMap[Symbol, Int]`

属性名からインデックス番号へのマッピング。

### `index2attr: HashMap[Int, Symbol]`

インデックス番号から属性名へのマッピング。

### `sparse_instances`

CSVデータをスパース表現に変換したデータ。各インスタンスは`(ArrayBuffer[(Symbol, Int)], Int)`の形式。

### `saveCsvFile(selected_attrs: List[Symbol], output_file_name: String)`

選択された属性のみを含む新しいCSVファイルを保存します。

## テスト

CSVReaderの動作は`src/test/scala/CSVReaderTest.scala`でテストされています。

```bash
sbt test
```

すべてのテストが成功することを確認してください。
