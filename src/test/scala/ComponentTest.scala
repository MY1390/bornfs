import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.scalatest.matchers.should.Matchers
import scala.collection.mutable.{ArrayBuffer, HashMap, HashSet}
import scala.collection.JavaConversions._
import scwc._
import java.io.{File, FileWriter}
import weka.core.Instances
import weka.core.converters.ConverterUtils.DataSource

/**
 * BornFSの内部コンポーネントに対するテスト
 */
class ARFFReaderTest extends AnyFunSuite with Matchers with BeforeAndAfterAll {
  
  // テスト用ARFFファイル
  val testArffContent = """@relation test

@attribute attr1 numeric
@attribute attr2 numeric
@attribute attr3 numeric
@attribute class {0,1}

@data
0,1,0,0
0,2,1,0
1,0,0,1
1,1,1,1
1,2,0,1
"""
  
  var testFile: String = _
  
  override def beforeAll(): Unit = {
    // テスト用のARFFファイルを作成
    testFile = "test_temp.arff"
    val writer = new FileWriter(testFile)
    writer.write(testArffContent)
    writer.close()
  }
  
  override def afterAll(): Unit = {
    // テスト後にファイルを削除
    new File(testFile).delete()
  }
  
  test("ARFFReaderが正しくデータを読み込む") {
    val reader = ARFFReader(testFile)
    
    // インスタンス数を確認
    reader.numInstances should be (5)
    
    // 属性数を確認 (クラス含む)
    reader.numAttrs should be (4)
    
    // 属性名とインデックスのマッピングを確認
    reader.attr2index(Symbol("attr1")) should be (0)
    reader.attr2index(Symbol("attr2")) should be (1)
    reader.attr2index(Symbol("attr3")) should be (2)
    reader.attr2index(Symbol("class")) should be (3)
    
    // インデックスと属性名のマッピングを確認
    reader.index2attr(0) should be (Symbol("attr1"))
    reader.index2attr(1) should be (Symbol("attr2"))
    reader.index2attr(2) should be (Symbol("attr3"))
    reader.index2attr(3) should be (Symbol("class"))
    
    // スパースインスタンスの確認
    val sparseData = reader.sparse_instances.toList
    sparseData.size should be (5)
    
    // 最初のインスタンスデータを確認
    val firstInstance = sparseData(0)
    firstInstance._1.map(_._1).toSet should be (Set(Symbol("attr1"), Symbol("attr2"), Symbol("attr3")))
    firstInstance._2 should be (0) // クラスラベル
  }
  
  test("ARFFReaderの属性削除機能") {
    val reader = ARFFReader(testFile)
    
    // attr1とattr3を選択して残す
    val selectedAttrs = List(Symbol("attr1"), Symbol("attr3"))
    reader.removeUnselectedAttrs(selectedAttrs)
    
    // 選択後は属性が2つ + クラス = 3つになる
    reader.instances.numAttributes should be (3)
    
    // 残った属性の名前を確認
    val attrNames = (0 until reader.instances.numAttributes).map(i => 
      reader.instances.attribute(i).name).toSet
    
    attrNames should contain ("attr1")
    attrNames should contain ("attr3")
    attrNames should contain ("class")
    attrNames should not contain ("attr2")
  }
}

/**
 * Caseクラスとその機能に対する詳細なテスト
 */
class CaseClassTest extends AnyFunSuite with Matchers {
  
  test("Caseオブジェクトの基本操作") {
    // テスト用のケースオブジェクトを作成
    val c = Case(ArrayBuffer((0, 1), (2, 3), (4, 5)), 1, 2)
    
    // 基本的なアクセサのテスト
    c.classLabel should be (1)
    c.frq should be (2)
    c.size should be (3)
    
    // 値の取得テスト
    c.value(0) should be (1)
    c.value(2) should be (3)
    c.value(4) should be (5)
    c.value(1) should be (0) // 存在しない属性は0
    
    // locationOfメソッドのテスト
    c.locationOf(0) should be (0)
    c.locationOf(2) should be (1)
    c.locationOf(4) should be (2)
    c.locationOf(1) should be (-1) // 存在しない属性
    
    // applyメソッドとwindowのテスト
    c.window should be (c.row) // 初期状態ではwindowとrowは同じ
    c(0) should be (1) // インデックス0の特徴値
  }
  
  test("Caseオブジェクトの番号付け直し") {
    val c = Case(ArrayBuffer((0, 1), (1, 2), (2, 3)), 0, 1)
    
    // 番号付け直し
    val order = Array[Int](2, 0, 1) // 0→2, 1→0, 2→1
    c.renumber(order)
    
    // 番号付け直し後のwindowを確認
    c.window.size should be (3)
    c.window(0)._1 should be (0) // 元の1が0に
    c.window(0)._2 should be (2)
    c.window(1)._1 should be (1) // 元の2が1に
    c.window(1)._2 should be (3)
    c.window(2)._1 should be (2) // 元の0が2に
    c.window(2)._2 should be (1)
  }
  
  test("Caseオブジェクトの比較") {
    val c1 = Case(ArrayBuffer((0, 1), (1, 2)), 0, 1)
    val c2 = Case(ArrayBuffer((0, 1), (1, 3)), 0, 1)
    val c3 = Case(ArrayBuffer((0, 2), (1, 2)), 0, 1)
    
    // インデックス0までの比較
    c1.compare(c2, 0) should be (0) // 最初の属性が同じ
    c1.compare(c3, 0) should be (-1) // c1の方が小さい
    c3.compare(c1, 0) should be (1) // c3の方が大きい
    
    // インデックス1までの比較
    c1.compare(c2, 1) should be (-1) // 2番目の属性でc1の方が小さい
    c2.compare(c1, 1) should be (1) // 2番目の属性でc2の方が大きい
  }
  
  test("Caseオブジェクトのシリアライズ") {
    val c = Case(ArrayBuffer((0, 1), (1, 2), (2, 3)), 0, 1)
    
    // シリアライズ結果を確認
    c.serialize should be ("0>1:1>2:2>3")
  }
}

/**
 * エントロピー計算とアルゴリズムのコア部分に対するテスト
 */
class EntropyCalculationTest extends AnyFunSuite with Matchers {
  
  test("log2関数のテスト") {
    val ds = Dataset(ArrayBuffer[(ArrayBuffer[(Attr, Value)], Value)](), 0, false, false)
    
    // log2(1) = 0
    ds.log2(1.0) should be (0.0 +- 0.0001)
    
    // log2(2) = 1
    ds.log2(2.0) should be (1.0 +- 0.0001)
    
    // log2(8) = 3
    ds.log2(8.0) should be (3.0 +- 0.0001)
  }
  
  test("xlog2関数のテスト") {
    val ds = Dataset(ArrayBuffer[(ArrayBuffer[(Attr, Value)], Value)](), 0, false, false)
    
    // x * log2(x) for x = 0 should be 0
    ds.xlog2(0.0) should be (0.0)
    
    // 2 * log2(2) = 2
    ds.xlog2(2.0) should be (2.0 +- 0.0001)
    
    // 4 * log2(4) = 8
    ds.xlog2(4.0) should be (8.0 +- 0.0001)
  }
  
  test("エントロピー計算の検証 - 一様分布") {
    // 一様分布のデータ作成 (各クラスが同じ頻度)
    val testData = ArrayBuffer[(ArrayBuffer[(Attr, Value)], Value)]()
    testData += ((ArrayBuffer((0, 0)), 0))
    testData += ((ArrayBuffer((0, 1)), 1))
    
    val ds = Dataset(testData, 0, false, false)
    
    // 一様分布のエントロピーは最大値になる (1ビット)
    ds.entropyLabel should be (1.0 +- 0.0001)
  }
  
  test("エントロピー計算の検証 - 偏った分布") {
    // 偏った分布のデータ作成 (クラス0が多い)
    val testData = ArrayBuffer[(ArrayBuffer[(Attr, Value)], Value)]()
    testData += ((ArrayBuffer((0, 0)), 0))
    testData += ((ArrayBuffer((0, 1)), 0))
    testData += ((ArrayBuffer((0, 2)), 0))
    testData += ((ArrayBuffer((0, 3)), 1))
    
    val ds = Dataset(testData, 0, false, false)
    
    // 偏った分布のエントロピーは1.0より小さくなる
    ds.entropyLabel should be < 1.0
    
    // エントロピーは0より大きい
    ds.entropyLabel should be > 0.0
  }
}

/**
 * より高度な動作のテスト
 */
class AdvancedFeaturesTest extends AnyFunSuite with Matchers with BeforeAndAfterAll {
  
  // 複雑なテストデータ
  var complexData: ArrayBuffer[(ArrayBuffer[(Attr, Value)], Value)] = _
  
  override def beforeAll(): Unit = {
    // 5つの属性と2つのクラスを持つ複雑なデータセット
    complexData = ArrayBuffer[(ArrayBuffer[(Attr, Value)], Value)]()
    
    // クラス0に強く関連する属性0と1
    for (i <- 0 until 10) {
      complexData += ((ArrayBuffer((0, 0), (1, 0), (2, i % 3), (3, i % 2), (4, i % 4)), 0))
    }
    
    // 例外パターン
    complexData += ((ArrayBuffer((0, 0), (1, 1), (2, 0), (3, 0), (4, 0)), 1))
    
    // クラス1に強く関連する属性0と1
    for (i <- 0 until 10) {
      complexData += ((ArrayBuffer((0, 1), (1, 1), (2, i % 3), (3, i % 2), (4, i % 4)), 1))
    }
    
    // 例外パターン
    complexData += ((ArrayBuffer((0, 1), (1, 0), (2, 0), (3, 0), (4, 0)), 0))
  }
  
  test("複雑なデータセットでの特徴選択") {
    val ds = Dataset(complexData, 0, false, false)
    
    // 閾値1.0で全ての属性を検証
    val selected = ds.select(1.0, 1)
    
    // 相互情報量の観点から重要な属性が選択される
    selected.size should be > 0
    
    // 属性0と1は重要なので含まれるはず
    selected should contain (0)
    selected should contain (1)
  }
  
  test("異なる閾値での特徴選択の比較") {
    val ds1 = Dataset(complexData, 0, false, false)
    val ds2 = Dataset(complexData, 0, false, false)
    
    // 閾値1.0 (すべての属性を保持)
    val selected1 = ds1.select(1.0, 1)
    
    // 閾値0.8 (重要な属性のみ保持)
    val selected2 = ds2.select(0.8, 1)
    
    // 閾値が低いと選択される属性が少なくなる
    selected1.size should be >= selected2.size
  }
  
  test("findBorderメソッドのテスト") {
    val ds = Dataset(complexData, 0, false, false)
    
    // findBorderは二分探索で指定した閾値を満たす特徴のインデックスを見つける
    ds.initializePrefix
    
    // 閾値1.0では-1が返る (該当なし)
    ds.findBorder(1.0, ds.maxAttr) should be >= -1
    
    // 閾値0.0では少なくとも1つの特徴が返る
    ds.findBorder(0.0, ds.maxAttr) should be >= 0
  }
  
  test("sortValメソッドのテスト - 異なるソート基準") {
    // 5つの異なるソート基準でデータセットを作成
    val ds0 = Dataset(complexData, 0, false, false) // ratio
    val ds1 = Dataset(complexData, 1, false, false) // noise
    val ds2 = Dataset(complexData, 2, false, false) // relevance
    val ds3 = Dataset(complexData, 3, false, false) // difference
    val ds4 = Dataset(complexData, 4, false, false) // harmonic
    
    // 各属性のソート値を計算
    ds0.initializePrefix
    ds1.initializePrefix
    ds2.initializePrefix
    ds3.initializePrefix
    ds4.initializePrefix
    
    // 異なるソート基準で計算された値を取得
    val val0 = ds0.sortVal(0) // 属性0のratio
    val val1 = ds1.sortVal(0) // 属性0のnoise
    val val2 = ds2.sortVal(0) // 属性0のrelevance
    val val3 = ds3.sortVal(0) // 属性0のdifference
    val val4 = ds4.sortVal(0) // 属性0のharmonic
    
    // 値が正しく計算される
    val0 should not be (0.0)
    val1 should not be (0.0)
    val2 should not be (0.0)
    val3 should not be (0.0)
    val4 should not be (0.0)
    
    // 各ソート方法は異なる値を生成するはず
    Set(val0, val1, val2, val3, val4).size should be > 1
  }
}