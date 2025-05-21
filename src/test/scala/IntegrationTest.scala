import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.scalatest.matchers.should.Matchers
import scala.collection.mutable.ArrayBuffer
import java.io.{File, PrintWriter}
import java.nio.file.{Files, Paths}

class ARFFReaderIntegrationTest extends AnyFunSuite with Matchers with BeforeAndAfterAll {
  
  val testDataPath = "data/test_integration.arff"
  val testOutputPath = "data/test_output.arff"
  
  override def beforeAll(): Unit = {
    // Create test ARFF file
    createTestARFFFile()
  }
  
  override def afterAll(): Unit = {
    // Clean up test files
    cleanupTestFiles()
  }
  
  def createTestARFFFile(): Unit = {
    val testData = """@relation test-dataset
@attribute feature1 numeric
@attribute feature2 numeric
@attribute feature3 numeric
@attribute feature4 numeric
@attribute class {0,1,2}

@data
1,0,3,2,0
0,2,1,0,1
2,1,0,3,2
1,1,2,1,0
0,0,1,2,1
3,2,0,1,2
""".stripMargin

    val writer = new PrintWriter(new File(testDataPath))
    writer.write(testData)
    writer.close()
  }
  
  def cleanupTestFiles(): Unit = {
    List(testDataPath, testOutputPath).foreach { path =>
      if (Files.exists(Paths.get(path))) {
        Files.delete(Paths.get(path))
      }
    }
  }
  
  test("ARFFReader should correctly load ARFF file") {
    val reader = ARFFReader(testDataPath)
    
    reader.numInstances should be(6)
    reader.numAttrs should be(5) // 4 features + 1 class
    
    // Test attribute mapping
    reader.attr2index should contain key Symbol("feature1")
    reader.attr2index should contain key Symbol("feature2")
    reader.attr2index should contain key Symbol("feature3")
    reader.attr2index should contain key Symbol("feature4")
    reader.attr2index should contain key Symbol("class")
    
    reader.attr2index(Symbol("feature1")) should be(0)
    reader.attr2index(Symbol("class")) should be(4)
  }
  
  test("ARFFReader should correctly convert to sparse instances") {
    val reader = ARFFReader(testDataPath)
    val sparseInstances = reader.sparse_instances.toList
    
    sparseInstances should have size 6
    
    // Check first instance: 1,0,3,2,0
    val firstInstance = sparseInstances.head
    val features = firstInstance._1
    val classLabel = firstInstance._2
    
    classLabel should be(0)
    features should contain((Symbol("feature1"), 1))
    features should contain((Symbol("feature3"), 3))
    features should contain((Symbol("feature4"), 2))
    // feature2 should not be present since it's 0 (sparse representation)
    features.exists(_._1 == Symbol("feature2")) should be(false)
  }
  
  test("ARFFReader should correctly remove unselected attributes") {
    val reader = ARFFReader(testDataPath)
    val originalAttrs = reader.numAttrs
    
    val selectedAttrs = List(Symbol("feature1"), Symbol("feature3"))
    reader.removeUnselectedAttrs(selectedAttrs)
    
    // Should have 2 selected features + 1 class attribute
    reader.instances.numAttributes should be(3)
    reader.instances.numInstances should be(6)
  }
  
  test("ARFFReader should correctly save ARFF file") {
    val reader = ARFFReader(testDataPath)
    val selectedAttrs = List(Symbol("feature1"), Symbol("feature2"))
    
    reader.removeUnselectedAttrs(selectedAttrs)
    reader.saveArffFile(testOutputPath)
    
    Files.exists(Paths.get(testOutputPath)) should be(true)
    
    // Verify the saved file can be read back
    val reloadedReader = ARFFReader(testOutputPath)
    reloadedReader.numAttrs should be(3) // 2 selected + 1 class
    reloadedReader.numInstances should be(6)
  }
}

class BornFSAlgorithmIntegrationTest extends AnyFunSuite with Matchers with BeforeAndAfterAll {
  
  val testDataPath = "data/bornfs_test.arff"
  var testReader: ARFFReader = _
  var testDataset: Dataset = _
  
  override def beforeAll(): Unit = {
    createBornFSTestData()
    testReader = ARFFReader(testDataPath)
    val mapdata = testReader.sparse_instances.to[ArrayBuffer].map { x =>
      (x._1.map { y => (testReader.attr2index(y._1), y._2) }, x._2)
    }
    testDataset = Dataset(mapdata, sort = 0, tutorial = false, verbose = false)
  }
  
  override def afterAll(): Unit = {
    if (Files.exists(Paths.get(testDataPath))) {
      Files.delete(Paths.get(testDataPath))
    }
  }
  
  def createBornFSTestData(): Unit = {
    val testData = """@relation bornfs-test
@attribute attr1 numeric
@attribute attr2 numeric
@attribute attr3 numeric
@attribute attr4 numeric
@attribute attr5 numeric
@attribute class {0,1}

@data
1,0,1,0,1,0
0,1,0,1,0,1
1,1,0,0,1,0
0,0,1,1,0,1
1,0,0,1,0,0
0,1,1,0,1,1
""".stripMargin

    val writer = new PrintWriter(new File(testDataPath))
    writer.write(testData)
    writer.close()
  }
  
  test("Dataset should correctly initialize with sparse data") {
    testDataset.maxAttr should be >= 0
    testDataset.maxLabel should be >= 0
    testDataset.nCases should be > 0
    testDataset.nSamples should be > 0
    
    testDataset.entropyLabel should be > 0.0
    testDataset.entropyEntire should be > 0.0
    testDataset.miEntireLabel should be >= 0.0
  }
  
  test("Dataset entropy calculations should be consistent") {
    // H(C) should be positive for non-deterministic data
    testDataset.entropyLabel should be > 0.0
    
    // H(Entire) should be >= H(C)
    testDataset.entropyEntire should be >= testDataset.entropyLabel
    
    // I(Entire;C) should be non-negative
    testDataset.miEntireLabel should be >= 0.0
    
    // I(Entire;C) = H(Entire) + H(C) - H(Entire,C)
    val computedMI = testDataset.entropyEntire + testDataset.entropyLabel - testDataset.entropyEntireLabel
    math.abs(testDataset.miEntireLabel - computedMI) should be < 1e-10
  }
  
  test("BornFS feature selection with different thresholds") {
    val thresholds = List(0.1, 0.5, 1.0)
    val hop = 1
    
    thresholds.foreach { threshold =>
      val selectedFeatures = testDataset.select(threshold, hop)
      
      // Should select at least one feature for reasonable thresholds
      if (threshold <= 1.0) {
        selectedFeatures should not be empty
      }
      
      // All selected features should be valid indices
      selectedFeatures.foreach { feature =>
        feature should be >= 0
        feature should be <= testDataset.maxAttr
      }
      
      // Should not select duplicate features
      selectedFeatures.distinct should have size selectedFeatures.size
    }
  }
  
  test("BornFS feature selection with different sorting methods") {
    val sortMethods = List(0, 1, 2, 3, 4) // ratio, noise, relevance, difference, harmonic
    val threshold = 0.8
    val hop = 1
    
    sortMethods.foreach { sortMethod =>
      val mapdata = testReader.sparse_instances.to[ArrayBuffer].map { x =>
        (x._1.map { y => (testReader.attr2index(y._1), y._2) }, x._2)
      }
      val dataset = Dataset(mapdata, sortMethod, tutorial = false, verbose = false)
      val selectedFeatures = dataset.select(threshold, hop)
      
      selectedFeatures should not be empty
      selectedFeatures.foreach { feature =>
        feature should be >= 0
        feature should be <= dataset.maxAttr
      }
    }
  }
  
  test("BornFS should handle edge cases gracefully") {
    val mapdata = testReader.sparse_instances.to[ArrayBuffer].map { x =>
      (x._1.map { y => (testReader.attr2index(y._1), y._2) }, x._2)
    }
    
    // Test with very high threshold (should select few or no features)
    val datasetHighThreshold = Dataset(mapdata, sort = 0, tutorial = false, verbose = false)
    val highThresholdResult = datasetHighThreshold.select(10.0, 1)
    // With very high threshold, might select no additional features beyond minimum
    
    // Test with very low threshold (should select more features)
    val datasetLowThreshold = Dataset(mapdata, sort = 0, tutorial = false, verbose = false)
    val lowThresholdResult = datasetLowThreshold.select(0.01, 1)
    lowThresholdResult should not be empty
    
    // Test with hop = 0 (sort only once)
    val datasetHopZero = Dataset(mapdata, sort = 0, tutorial = false, verbose = false)
    val hopZeroResult = datasetHopZero.select(0.5, 0)
    hopZeroResult should not be empty
  }
  
  test("Dataset performance metrics should be recorded") {
    val mapdata = testReader.sparse_instances.to[ArrayBuffer].map { x =>
      (x._1.map { y => (testReader.attr2index(y._1), y._2) }, x._2)
    }
    val dataset = Dataset(mapdata, sort = 0, tutorial = false, verbose = false)
    
    // Execute feature selection
    dataset.select(0.5, 1)
    
    // Check that timing information is recorded
    dataset.timeSortFeatures should be >= 0L
    dataset.timeSortInstances should be >= 0L
    dataset.timeSelectFeatures should be >= 0L
    
    // Check that events are recorded
    dataset.events should not be empty
    dataset.events.foreach { event =>
      event._1 should not be null
      event._2 should not be empty
    }
  }
}

class EndToEndIntegrationTest extends AnyFunSuite with Matchers with BeforeAndAfterAll {
  
  val inputPath = "data/e2e_input.arff"
  val outputPath = "data/e2e_output.arff"
  
  override def beforeAll(): Unit = {
    createEndToEndTestData()
  }
  
  override def afterAll(): Unit = {
    List(inputPath, outputPath).foreach { path =>
      if (Files.exists(Paths.get(path))) {
        Files.delete(Paths.get(path))
      }
    }
  }
  
  def createEndToEndTestData(): Unit = {
    val testData = """@relation end-to-end-test
@attribute f1 numeric
@attribute f2 numeric
@attribute f3 numeric
@attribute f4 numeric
@attribute f5 numeric
@attribute f6 numeric
@attribute target {A,B,C}

@data
1,2,0,1,3,0,A
0,1,2,0,1,3,B
2,0,1,3,0,1,C
1,1,1,1,1,1,A
0,0,2,2,2,2,B
3,3,0,0,0,0,C
1,0,2,1,0,2,A
0,2,1,0,2,1,B
""".stripMargin

    val writer = new PrintWriter(new File(inputPath))
    writer.write(testData)
    writer.close()
  }
  
  test("Complete end-to-end feature selection workflow") {
    // Step 1: Load data
    val reader = ARFFReader(inputPath)
    reader.numInstances should be(8)
    reader.numAttrs should be(7) // 6 features + 1 class
    
    // Step 2: Convert to sparse format
    val sparseData = reader.sparse_instances.to[ArrayBuffer].map { instance =>
      (instance._1.map { feature => 
        (reader.attr2index(feature._1), feature._2) 
      }, instance._2)
    }
    
    // Step 3: Create dataset and run feature selection
    val dataset = Dataset(sparseData, sort = 0, tutorial = false, verbose = false)
    val selectedFeatures = dataset.select(delta = 0.6, hop = 1)
    
    // Step 4: Verify selection results
    selectedFeatures should not be empty
    selectedFeatures.size should be <= 6 // Cannot select more than available features
    
    // Step 5: Map back to feature names
    val selectedAttrNames = selectedFeatures.map { idx => 
      reader.index2attr(idx) 
    }.toList
    
    selectedAttrNames should not be empty
    selectedAttrNames.foreach { attrName =>
      reader.attr2index should contain key attrName
    }
    
    // Step 6: Create filtered dataset
    reader.removeUnselectedAttrs(selectedAttrNames)
    reader.instances.numAttributes should be(selectedAttrNames.size + 1) // +1 for class
    
    // Step 7: Save filtered dataset
    reader.saveArffFile(outputPath)
    Files.exists(Paths.get(outputPath)) should be(true)
    
    // Step 8: Verify saved file integrity
    val reloadedReader = ARFFReader(outputPath)
    reloadedReader.numInstances should be(8) // Same number of instances
    reloadedReader.numAttrs should be(selectedAttrNames.size + 1)
  }
  
  test("Feature selection quality assessment") {
    val reader = ARFFReader(inputPath)
    val sparseData = reader.sparse_instances.to[ArrayBuffer].map { instance =>
      (instance._1.map { feature => 
        (reader.attr2index(feature._1), feature._2) 
      }, instance._2)
    }
    
    val dataset = Dataset(sparseData, sort = 0, tutorial = false, verbose = false)
    
    // Test different thresholds and compare results
    val thresholds = List(0.2, 0.5, 0.8)
    val results = thresholds.map { threshold =>
      val freshDataset = Dataset(sparseData, sort = 0, tutorial = false, verbose = false)
      val features = freshDataset.select(threshold, 1)
      (threshold, features.size, features)
    }
    
    // Higher thresholds should generally select fewer features
    results.sliding(2).foreach { case List((t1, size1, _), (t2, size2, _)) =>
      // This is a general trend, but not strict due to algorithm behavior
      info(s"Threshold $t1 selected $size1 features, threshold $t2 selected $size2 features")
    }
    
    // All results should be valid
    results.foreach { case (threshold, size, features) =>
      size should be >= 0
      size should be <= 6
      features.foreach { feature =>
        feature should be >= 0
        feature should be <= dataset.maxAttr
      }
    }
  }
  
  test("Performance benchmarking") {
    val reader = ARFFReader(inputPath)
    val sparseData = reader.sparse_instances.to[ArrayBuffer].map { instance =>
      (instance._1.map { feature => 
        (reader.attr2index(feature._1), feature._2) 
      }, instance._2)
    }
    
    val startTime = System.nanoTime()
    val dataset = Dataset(sparseData, sort = 0, tutorial = false, verbose = false)
    val creationTime = System.nanoTime() - startTime
    
    val selectionStartTime = System.nanoTime()
    val selectedFeatures = dataset.select(0.5, 1)
    val selectionTime = System.nanoTime() - selectionStartTime
    
    // Basic performance assertions
    creationTime should be > 0L
    selectionTime should be > 0L
    
    // Verify internal timing measurements
    dataset.timeSortFeatures should be >= 0L
    dataset.timeSortInstances should be >= 0L
    dataset.timeSelectFeatures should be >= 0L
    
    val totalInternalTime = dataset.timeSortFeatures + 
                          dataset.timeSortInstances + 
                          dataset.timeSelectFeatures
    
    // Internal timing should be reasonable compared to external measurement
    totalInternalTime should be <= selectionTime * 2 // Allow some overhead
    
    info(s"Dataset creation: ${creationTime / 1000000}ms")
    info(s"Feature selection: ${selectionTime / 1000000}ms")
    info(s"Selected ${selectedFeatures.size} features")
  }
}