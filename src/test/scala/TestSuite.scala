import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.scalatest.matchers.should.Matchers
import scala.collection.mutable.{ArrayBuffer, HashMap}
import scwc._
import java.io.File
/**
 * BornFSアルゴリズムの包括的なテストスイート
 */
class BornFSTest extends AnyFunSuite with Matchers with BeforeAndAfterAll {
  
  // 共通のテストデータ
  val testData = ArrayBuffer[(ArrayBuffer[(Attr, Value)], Value)]()
  
  // テストデータの準備
  override def beforeAll(): Unit = {
    // 単純なデータセットを作成
    // 属性1: 0か1の値 (関連あり)
    // 属性2: 0,1,2の値 (関連あり)
    // 属性3: 0,1の値 (関連なし - ランダム)
    // クラス: 0か1
    
    // 属性1が0ならクラスは0が多い
    testData += ((ArrayBuffer((0, 0), (1, 0), (2, 0)), 0))
    testData += ((ArrayBuffer((0, 0), (1, 1), (2, 1)), 0))
    testData += ((ArrayBuffer((0, 0), (1, 2), (2, 0)), 0))
    testData += ((ArrayBuffer((0, 0), (1, 0), (2, 1)), 0))
    testData += ((ArrayBuffer((0, 0), (1, 1), (2, 0)), 1)) // 例外パターン
    
    // 属性1が1ならクラスは1が多い
    testData += ((ArrayBuffer((0, 1), (1, 0), (2, 1)), 1))
    testData += ((ArrayBuffer((0, 1), (1, 1), (2, 0)), 1))
    testData += ((ArrayBuffer((0, 1), (1, 2), (2, 1)), 1))
    testData += ((ArrayBuffer((0, 1), (1, 2), (2, 0)), 1))
    testData += ((ArrayBuffer((0, 1), (1, 0), (2, 0)), 0)) // 例外パターン
  }
  
  test("エントロピー計算の基本テスト") {
    // 属性の選択なし、閾値1.0でデータセットを作成
    val ds = Dataset(testData, 0, false, false)
    
    // エントロピー値は0より大きい
    ds.entropyLabel should be > 0.0
    ds.entropyEntire should be > 0.0
    ds.entropyEntireLabel should be > 0.0
    
    // クラスラベルのエントロピーは理論上の最大値以下
    ds.entropyLabel should be <= 1.0 // 2クラス問題の最大エントロピーは1.0
  }
  
  test("相互情報量の基本検証") {
    val ds = Dataset(testData, 0, false, false)
    
    // 相互情報量は非負
    ds.miEntireLabel should be >= 0.0
    
    // 属性1と2は強い関連があるはず
    val attr0MI = ds.entropyAttr(0) + ds.entropyLabel - ds.entropyAttrLabel(0)
    val attr1MI = ds.entropyAttr(1) + ds.entropyLabel - ds.entropyAttrLabel(1)
    val attr2MI = ds.entropyAttr(2) + ds.entropyLabel - ds.entropyAttrLabel(2)
    
    // 属性3はランダムなので、相互情報量は低いはず
    attr0MI should be > attr2MI
    attr1MI should be > attr2MI
  }
  
  test("特徴選択の動作検証 - 閾値1.0") {
    val ds = Dataset(testData, 0, false, false)
    
    // 閾値1.0で全ての特徴が選択される
    val selectedFeatures = ds.select(1.0, 1)
    
    // 特徴が選択されている
    selectedFeatures.size should be > 0
    
    // 選択された特徴を確認
    selectedFeatures.foreach { attr =>
      attr should be >= 0
      attr should be <= 2
    }
  }
  
  test("特徴選択の動作検証 - 閾値0.95") {
    val ds = Dataset(testData, 0, false, false)
    
    // 閾値0.95では関連性の高い特徴のみが選択される
    val selectedFeatures = ds.select(0.95, 1)
    
    // 特徴が選択されている
    selectedFeatures.size should be > 0
    selectedFeatures.size should be < 3 // 全部は選ばれない
    
    // 関連性の低い属性3は選ばれないはず
    selectedFeatures should not contain 2
  }
  
  test("特徴ソートのテスト") {
    val ds = Dataset(testData, 0, false, false)
    
    // 元のインデックスと特徴を参照
    val entity = (0 to 2).toArray
    
    // 相互情報量に基づいて属性をソート
    ds.initializePrefix
    val (sorted, order) = ds.sortAttrs
    
    // ソートされた結果は3つの属性
    sorted.length should be (3)
    
    // ソートされた順序を確認 (相互情報量が高い順)
    // 属性0と1は関連性が高く、属性2は関連性が低い
    sorted.indexOf(2) should be > sorted.indexOf(0)
    sorted.indexOf(2) should be > sorted.indexOf(1)
  }
  
  test("異なるソート基準でのテスト") {
    // ratio (0), noise (1), relevance (2), difference (3), harmonic (4)
    val dsRatio = Dataset(testData, 0, false, false)
    val dsNoise = Dataset(testData, 1, false, false)
    val dsRelevance = Dataset(testData, 2, false, false)
    
    // 各ソート方法で特徴選択を実行
    val selectedRatio = dsRatio.select(0.9, 1)
    val selectedNoise = dsNoise.select(0.9, 1)
    val selectedRelevance = dsRelevance.select(0.9, 1)
    
    // 各方法で少なくとも1つの特徴が選択される
    selectedRatio.size should be > 0
    selectedNoise.size should be > 0
    selectedRelevance.size should be > 0
  }
  
  test("Case クラスの機能テスト") {
    // Caseオブジェクトのテスト
    val c1 = Case(ArrayBuffer((0, 0), (1, 1)), 0, 1)
    val c2 = Case(ArrayBuffer((0, 0), (1, 2)), 1, 1)
    
    // value メソッドのテスト
    c1.value(0) should be (0)
    c1.value(1) should be (1)
    c1.value(2) should be (0) // 存在しない属性は0
    
    // indexOf メソッドのテスト
    c1.indexOf(0) should be >= 0
    c1.indexOf(5) should be (-1) // 存在しないインデックス
    
    // compare メソッドのテスト - 辞書式順序
    c1.compare(c2, 0) should be (0) // 最初の属性が同じ
  }
  
  test("実際のARFFファイルでのテスト") {
    // テスト用のARFFファイルが存在する場合のみ実行
    val testFile = "data/test.arff"
    if (new File(testFile).exists()) {
      val reader = ARFFReader(testFile)
      val data = reader.sparse_instances.to[ArrayBuffer].map{x =>
        (x._1.map{y => (reader.attr2index(y._1), y._2)}, x._2)}
      
      val ds = Dataset(data, 0, false, false)
      val selected = ds.select(0.9, 1)
      
      // 少なくとも1つの特徴が選択される
      selected.size should be > 0
      
      // 選択された特徴の範囲チェック
      selected.foreach { attr =>
        attr should be >= 0
        attr should be < reader.numAttrs
      }
    } else {
      // ファイルがない場合はスキップ
      cancel("テストファイル data/test.arff が見つかりません")
    }
  }
  
  test("異なるhop値でのテスト") {
    val ds = Dataset(testData, 0, false, false)
    
    // hop=1 (毎回特徴をソート)
    val selected1 = ds.select(0.9, 1)
    
    // hop=0 (初回のみソート)
    val ds2 = Dataset(testData, 0, false, false)
    val selected2 = ds2.select(0.9, 0)
    
    // 少なくとも1つの特徴が選択される
    selected1.size should be > 0
    selected2.size should be > 0
  }
}