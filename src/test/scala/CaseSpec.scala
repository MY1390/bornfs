import munit.FunSuite
import java.io.{File, ByteArrayOutputStream, PrintStream}
import scala.collection.mutable.{ArrayBuffer, HashMap}
import weka.core.{Instances, Attribute}
import weka.core.converters.ConverterUtils.DataSource
import scwc._

class CaseSpec extends FunSuite {

  val sampleRow = ArrayBuffer[(Attr, Value)]((1, 10), (3, 30), (5, 50))
  val sampleCase = Case(sampleRow.clone(), 1, 5)

  test("Case constructor create case with correct properties") {
    assertEquals(sampleCase.classLabel, 1)
    assertEquals(sampleCase.frq, 5)
    assertEquals(sampleCase.size, 3)
    assertEquals(sampleCase.window.toSet, sampleRow.toSet)
  }

  test("apply method return correct value for existing index") {
    assertEquals(sampleCase(0), 10)  // First element in window
    assertEquals(sampleCase(1), 30)  // Second element in window
    assertEquals(sampleCase(2), 50)  // Third element in window
  }

  test("it should return 0 for non-existing index") {
    assertEquals(sampleCase(10), 0)
    assertEquals(sampleCase(-1), 0)
  }

  test("indexOf method return correct position for existing index") {
    assertEquals(sampleCase.indexOf(1), 0)
    assertEquals(sampleCase.indexOf(3), 1)
    assertEquals(sampleCase.indexOf(5), 2)
  }

  test("it should return -1 for non-existing index") {
    assertEquals(sampleCase.indexOf(0), -1)
    assertEquals(sampleCase.indexOf(2), -1)
    assertEquals(sampleCase.indexOf(10), -1)
  }

  test("value method should return correct value for existing attribute") {
    assertEquals(sampleCase.value(1), 10)
    assertEquals(sampleCase.value(3), 30)
    assertEquals(sampleCase.value(5), 50)
  }

  test("it should return 0 for non-existing attribute") {
    assertEquals(sampleCase.value(0), 0)
    assertEquals(sampleCase.value(2), 0)
    assertEquals(sampleCase.value(10), 0)
  }

  test("locationOf method should return correct location for existing attribute") {
    assertEquals(sampleCase.locationOf(1), 0)
    assertEquals(sampleCase.locationOf(3), 1)
    assertEquals(sampleCase.locationOf(5), 2)
  }

  test("it should return -1 for non-existing attribute") {
    assertEquals(sampleCase.locationOf(0), -1)
    assertEquals(sampleCase.locationOf(2), -1)
  }

  test("renumber method should correctly renumber window indices") {
    val order = Array(2, 0, 1)
    sampleCase.renumber(order)
    
    // After renumbering, the indices should be reordered
    assertEquals(sampleCase.window.head._1, 0)  // was index 1, now index 0
    assertEquals(sampleCase.window(1)._1, 1)     // was index 3, now index 1  
    assertEquals(sampleCase.window(2)._1, 2)     // was index 5, now index 2
  }

  test("compare method should return correct comparison result") {
    val case1 = Case(ArrayBuffer((1, 10), (2, 20)), 1, 1)
    val case2 = Case(ArrayBuffer((1, 10), (2, 25)), 1, 1)
    val case3 = Case(ArrayBuffer((1, 10), (2, 20)), 1, 1)
    
    assert(case1.compare(case2, 2) < 0)  // case1 < case2
    assert(case2.compare(case1, 2) > 0)  // case2 > case1
    assertEquals(case1.compare(case3, 2), 0)     // case1 == case3
  }

  test("serialize method return correct string representation") {
    val serialized = sampleCase.serialize
    assert(serialized.contains("1>10"))
    assert(serialized.contains("3>30"))
    assert(serialized.contains("5>50"))
    assert(serialized.contains(":"))
  }
}