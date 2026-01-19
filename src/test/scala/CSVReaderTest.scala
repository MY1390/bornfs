import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.scalatest.matchers.should.Matchers
import scala.collection.mutable.ArrayBuffer

/**
 * CSVReaderのテストクラス
 * test.csvファイルを読み込んで、正しく解析できることを確認
 */
class CSVReaderTest extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  var reader: CSVReader = _

  override def beforeAll(): Unit = {
    val path: String = "data/test.csv"
    reader = CSVReader(path)
    println("=== CSVReader Test ===")
    println(s"読み込んだインスタンス数: ${reader.numInstances}")
    println(s"属性数: ${reader.numAttrs}")
  }

  test("CSVファイルが正しく読み込まれる") {
    reader.numInstances shouldBe 10
    reader.numAttrs shouldBe 7
  }

  test("属性名とインデックスのマッピングが正しい") {
    reader.attr2index(Symbol("a")) shouldBe 0
    reader.attr2index(Symbol("b")) shouldBe 1
    reader.attr2index(Symbol("c")) shouldBe 2
    reader.attr2index(Symbol("class")) shouldBe 6
  }

  test("インデックスと属性名のマッピングが正しい") {
    reader.index2attr(0).name shouldBe "a"
    reader.index2attr(1).name shouldBe "b"
    reader.index2attr(6).name shouldBe "class"
  }

  test("スパース表現への変換が正しい") {
    val sparse = reader.sparse_instances
    sparse.length shouldBe 10

    // 最初のインスタンスをチェック: 0,1,0,0,1,0,0
    val first = sparse.head
    val first_attrs = first._1.toList
    val first_class = first._2

    // 値が1の属性は b(1) と e(4)
    first_attrs.map(_._1) should contain (Symbol("b"))
    first_attrs.map(_._1) should contain (Symbol("e"))
    first_class shouldBe 0
  }

  test("スパース表現の各インスタンスのクラスラベルが正しい") {
    val sparse = reader.sparse_instances
    val class_labels = sparse.map(_._2).toList

    // test.csvの各行のクラスラベル
    class_labels shouldBe List(0, 1, 0, 1, 1, 0, 1, 0, 0, 1)
  }
}
