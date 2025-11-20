import scwc._

import weka.core.Instances
import weka.core.converters.ArffSaver
import weka.core.converters.ConverterUtils.DataSource
import weka.filters.Filter
import weka.filters.unsupervised.attribute.Remove

import scala.collection.mutable.{HashMap,HashSet,ArrayBuffer}
import scala.collection.mutable
import scala.jdk.CollectionConverters._

import java.text.DecimalFormat
import java.io.{File, OutputStreamWriter, FileOutputStream}

import scopt.OptionParser

case class MainOption(
  threshold: Double = 1.0,
  hop: Int = 1,
  in: String = "",
  out: String = "",
  log: String = "low",
  sort: String = "ratio",
  tutorial: Boolean = false,
  verbose: Boolean = true
)

case class ARFFReader(filename: String) {
  var instances = new DataSource(filename).getDataSet
  // 属性名 -> インデックス番号のマッピング
  val attr2index = HashMap[Symbol,Int]()
  // インデックス番号 -> 属性名のマッピング
  val index2attr = HashMap[Int,Symbol]()
  // 全属性をイテレートして、属性名とインデックスの双方向マップを初期化
  // これにより、属性名からインデックス、インデックスから属性名への
  // 高速な変換が可能になる（O(1)のルックアップ）
  (0 until instances.numAttributes).foreach {index =>
    attr2index += Symbol(instances.attribute(index).name) -> index
    index2attr += index -> Symbol(instances.attribute(index).name)
  }
  // 特徴選択の対象となるデータセットのサイズを記録
  // numInstances: 学習データのサンプル数
  // numAttrs: 選択対象の特徴数（クラスラベル含む）
  val numInstances  = instances.numInstances
  val numAttrs = instances.numAttributes
  val sparse_instances = sparseInstances

  /**
   * ARFFデータをスパース表現に変換する
   * 0以外の値を持つ属性のみを(属性名, 値)のペアとして保持し、メモリ効率を向上
   * 
   * @return 各インスタンスを(非ゼロ属性のリスト, クラスラベル)のタプルとして返す
   */
  def sparseInstances = {
    instances.enumerateInstances().asScala.map { instance =>
      val n = instance.numValues
      // 最後の属性（通常クラスラベル）以外で、値が0でない属性を収集
      val body: ArrayBuffer[(Symbol, Int)] =
        for(i <- (0 until n - 1).to(mutable.ArrayBuffer)
          if instance.value(instance.index(i)).toInt != 0) yield {
          val attr_sym = Symbol(instance.attribute(instance.index(i)).name)
          (attr_sym, instance.value(instance.index(i)).toInt)
        }
      
      // 最後の属性の処理：クラスラベルかどうかを判定
      val temp_sym = Symbol(instance.attribute(instance.index(n - 1)).name)
      if(temp_sym == index2attr(instances.numAttributes - 1)) {
        // 最後の属性がクラスラベルの場合
        (body, instance.value(instance.index(n - 1)).toInt)
      } else {
        // 最後の属性もデータ属性の場合（クラスラベルなし）
        body += ((temp_sym, instance.value(instance.index(n - 1)).toInt))
        (body, 0)
      }
    }
  }

  // 特徴選択結果をARFFデータに反映：選択されなかった属性を削除
  def removeUnselectedAttrs(selected_attrs: List[Symbol]): Unit = {
    // 選択された属性のインデックスを取得し、クラスラベル（最後の属性）も追加
    val remove_list =
      ((for (attr <- selected_attrs) yield (attr2index(attr))) ::: List(instances.numAttributes - 1)).toArray

    val filter = new Remove()
    filter.setAttributeIndicesArray(remove_list)  // 削除対象として設定

    // setInvertSelection(true)により、削除対象を反転
    // つまり、remove_listの属性を「保持」し、それ以外を削除
    filter.setInvertSelection(true)
    filter.setInputFormat(instances)
    instances = Filter.useFilter(instances, filter)  // フィルタを適用してインスタンスを更新
  }

  def saveArffFile(output_file_name: String): Unit = {
    val arff_saver = new ArffSaver()
    arff_saver.setInstances(instances)
    arff_saver.setFile(new File(output_file_name))
    arff_saver.writeBatch()
  }
}

object Main {

// ログ出力用のフォーマッタ
  val f = new DecimalFormat("0.0000")      // H(X), I(X;Y), μ_H, μ_G等の表示
  val fns = new DecimalFormat("#,### nsec") // 各処理ステップの実行時間表示
  def main(args: Array[String]): Unit = {

    val parser = new OptionParser[MainOption]("Born Feature Selection") {
      opt[String]('i', "input").required().valueName("<path>").action { (x, o) =>
        o.copy(in = x)
      } text("A path to an input ARFF file without extension .arff")

      opt[String]('o', "output").valueName("<path>").action { (x, o) =>
        o.copy(out = x)
      } text("A path to an output ARFF file")

      opt[String]('s', "sort").valueName("<ratio, noise, relevance, difference, harmonic>").action { (x, o)  =>
        o.copy(sort = x)
      } text("A measure to sort features")

      opt[Double]('t', "threshold").valueName("<value>").action { (x, o) =>
        o.copy(threshold = x)
      } text("Threshold: 1.0 (default)")

      opt[Int]('h', "hop").valueName("<value>").action { (x, o) =>
        o.copy(hop = x)
      } text("Threshold: 1.0 (default)")

      opt[String]('l', "log").valueName("<high,low,none>").action { (x, o) =>
        o.copy(log = x)
      } text("Level of detail of log: high, low (default) or none")

      opt[Boolean]('T', "tutorial").valueName("<true,false>").action { (x, o) =>
        o.copy(tutorial = x)
      } text("Tutorial mode: true or false (default)")

      opt[Boolean]('v', "verbose").valueName("<true,false>").action { (x, o) =>
        o.copy(verbose = x)
      } text("Verbose mode: true (default) or false")
    }

    // コマンドラインパラメータのデフォルト値
    // アルゴリズムパラメータ
    var threshold = 1.0    // 特徴選択の閾値（相互情報量の比率）
    var hop = 1           // 特徴再ソートの頻度
    var sort = 0          // 特徴の評価基準（0: relevance/noise比）

    // ファイルI/O
    var in = ""           // 入力ARFFファイル
    var out = ""          // 出力ARFFファイル（オプション）

    // 実行モード
    var log = "low"       // ログ詳細度
    var tutorial = false  // チュートリアルモード
    var verbose = true    // 進捗表示

    // コマンドライン引数を解析してパラメータを設定
    parser.parse(args, MainOption()) match {
      case Some(option) =>
        // 必須パラメータ
        in = option.in
        out = option.out
        
        // アルゴリズムパラメータ
        threshold = option.threshold
        hop = option.hop
        
        // ソート基準を文字列から数値に変換
        sort = option.sort match {
          case "ratio" => 0      // relevance/noise比（推奨）
          case "noise" => 1      // ノイズゲインの負値
          case "relevance" => 2  // 関連性ゲイン
          case "difference" => 3 // 関連性 - ノイズ
          case "harmonic" => 4   // 調和平均
          case _ =>
            println("Wrong specification " + sort)
            println(parser.usage)
            return
        }
        
        // ログレベルの検証
        log = option.log
        log match {
          case "high" =>  // 詳細ログ（エントロピー値など含む）
          case "low"  =>  // 標準ログ（選択結果と基本統計）
          case "none" =>  // ログ出力なし
          case _ =>
            println("Wrong specification " + log)
            println(parser.usage)
            return
        }
        
        // 実行モード設定
        tutorial = option.tutorial
        verbose = option.verbose
        
      case None =>
        return  // 引数解析失敗時は終了
    }


    println("\n*** Born Feature Selection (Copy Rights 2019 K. Shin)\n")

    println("Input file = " + in)
    if(out != "") println("Output file = " + out)
    print("Sorting features is executed with respect to ")
    println(sort match {
      case 0 => "ratio of relevance to noise"
      case 1 => "noise"
      case 2 => "relevance"
      case 3 => "difference of relevance from noise"
      case 4 => "harmonic mean of relevance and noise"
    })
    println("The minimum ratio of mutual information of selected features to the maximum is " + threshold + ".")
    if(hop == 0) {
      println("Sorting features is executed once at the beginning.")
    } else {
      println("Sorting features is executed every " + hop + " iterations.")
    }
    println("Level of detail of log is " + log + ".")
    if(tutorial){
      println("The processing of feature selection will be explained step by step (tutorial mode on).")
    } else {
      println("Tutorial mode is off.")
    }
    if(verbose){
      println("States of execution will be displayed real time (verbose mode on).")
    } else {
      println("The program will run quietly (verbose mode off). ")
    }
    println

    print("Reading the file ... ")
    val db = ARFFReader(in)
    // スパースインスタンスをBornFS内部形式に変換
    val data = db.sparse_instances
      .to(mutable.ArrayBuffer)  // 可変配列に変換（処理効率のため）
      .map { x =>
        // x._1: 非ゼロ属性のリスト [(Symbol, Int)]
        // x._2: クラスラベル
        (
          x._1.map(y => (db.attr2index(y._1), y._2)),  // 属性名をインデックスに変換
          x._2  // クラスラベルはそのまま保持
        )
      }
      .toSeq  // 不変のシーケンスに変換
    println("finished.")
    println("Found "+db.numInstances+" instances and "+db.numAttrs+" features including class.")

    if(hop == 0) hop = db.numAttrs - 1

    if(tutorial && (db.numInstances > 10 || db.numAttrs > 20)) {
      println
      println("Tutorial mode is canceled because the dataset is too big (#Instance > 10 or #Feature > 20).")
      tutorial = false
    }

    if(!tutorial) print("Creating a dataset object...")

    val start = System.nanoTime()

    val ds = Dataset(data, sort, tutorial, verbose)

    val timeCreateObject = System.nanoTime() - start

    if(!tutorial) {
      println("finished.")
      println("Needed " + fns.format(timeCreateObject) + ".")
      print("Selecting features...")
    }

    val result = ds.select(threshold, hop)

    val selected_attrs = result.map{i => db.index2attr(i)}.toList

    if(!tutorial) {
      println("finished.")
      println("Needed " + fns.format(ds.timeSortFeatures +
        ds.timeSortInstances + ds.timeSelectFeatures) + ".")
    }

    // println("DEBUG>>")
    // println("Count the occurences of (attr_value, class_label)")
    // ds.ofc.foreach(x => println(x))
    // println("Count the occurences of attribute values")
    // ds.of.foreach(x => println(x))
    // println("Count the occurences of class labels")
    // ds.oc.foreach(x => println(x))
    // ds.displayStats(None)
    // println("The result of attribute sorting")
    // // val f = new DecimalFormat("0.00")
    // //   (0 until ds.rn.size).foreach {i =>
    // //     if(ds.nr(i) != HIDDEN)
    // //       print(ds.nr(i) + "(" + f.format(ds.msr(sl)(ds.nr(i))) + ") ")
    // //   }
    // // println
    // val f = new DecimalFormat("0.00")
    //   (0 until 30).foreach {i =>
    //     if(ds.nr(i) != HIDDEN)
    //       print(ds.nr(i) + "(" + f.format(ds.msr(sl)(ds.nr(i))) + ") ")
    //   }
    // println
    // println("<<DEBUG")

    println
    println(s"${result.size} features have been selected.")
    println
    println("Selected features are: " + selected_attrs.map(_.name).mkString(" "))

   // println("\nSelected attributes: feature name (score and rank in " + ds.slm(sort_selector(sort)) + ")")
   //  for (attr <- result) {
   //    print(f"$attr(${ds.msr(sort_selector(sort))(attr)}%.3f, ${ds.rn(attr) + 1}) ")
   //  }
    println
    /*
    println("Statistics")
    for(x <- (0 to 5) if x != sort_selector(sort)) {
      println("Scores in　" + ds.slm(x) + ":")
      for (attr <- result if attr!=HIDDEN) {
        print(f"$attr(${ds.msr(x)(attr)}%.3f) ")
      }
      println
      println
    }
     */

    if(out != "") {
      db.removeUnselectedAttrs(selected_attrs)
      db.saveArffFile(out)
      //db.saveArffFile(out + ".arff")
      println("The modified dataset has been output to " + out + ".\n")
    }

    if(log == "high" || log == "low") {
      val f = new DecimalFormat("0.0000")
      val log_file_name = in + (sort match{
        case 0 => "-ratio-"
        case 1 => "-noise-"
        case 2 => "-relevance-"
        case 3 => "-diff-"
        case 4 => "-muh-"
      }) + threshold + "-" + (if(hop >= db.numAttrs-1) 0 else hop) + "-bornfs.log"
      val log_file = new OutputStreamWriter(new FileOutputStream(log_file_name), "utf-8")
//      val log_file = new OutputStreamWriter(new FileOutputStream(inWOext+".log"), "utf-8")

      log_file.write("# Parameters\n")
      log_file.write("## Input file = " + in + "\n")
      if(out != "") log_file.write("## Output file = " + out + "\n")
      log_file.write("## Algorithm = bornfs\n")
      log_file.write("## Sorting measure = " + (sort match{
        case 0 => "ratio of relevance to noise\n"
        case 1 => "negative noise\n"
        case 2 => "relevance\n"
        case 3 => "difference of relevance from noise\n"
        case 4 => "harmonic mean of relevance and noise\n"
      })
      )
      log_file.write("## Threshold = " + threshold + "\n")
      log_file.write("## Hop = " + (if(hop >= db.numAttrs-1) 0 else hop) + "\n")
      log_file.write("## Level of detail of log = " + log + "\n")

      log_file.write("\n# Run-time\n")
      log_file.write("## Creating a dataset object = " + fns.format(timeCreateObject) + "\n")
      log_file.write("## Sorting features = " + fns.format(ds.timeSortFeatures) + "\n")
      log_file.write("## Sorting instances = " + fns.format(ds.timeSortInstances) + "\n")
      log_file.write("## Selecting features = " + fns.format(ds.timeSelectFeatures) + "\n")

      log_file.write("\n# " + selected_attrs.size + "features selected.\n")
      log_file.write("# Selected features\n")
      log_file.write(selected_attrs.map(_.name).mkString(""," ","\n"))

      log_file.write("\n# Statistics\n")
      log_file.write("## Number of instances = " + db.numInstances + "\n")
      log_file.write("## Number of features = " + db.numAttrs + "\n")
      val hc = ds.entropyLabel
      log_file.write("## H(C) = " + f.format(hc) + "\n")
      log_file.write("## H(Entire) = " + f.format(ds.entropyEntire) + "\n")
      log_file.write("## H(Entire, C) = " + f.format(ds.entropyEntireLabel) + "\n")
      log_file.write("## I(Entire; C) = " + f.format(ds.miEntireLabel) + "\n")
      log_file.write("## H(Selected) = " + f.format(ds.entropyPrefix) + "\n")
      log_file.write("## H(Selected, C) = " + f.format(ds.entropyPrefixLabel) + "\n")
      val mis = ds.entropyPrefix + hc - ds.entropyPrefixLabel
      log_file.write("## I(Selected; C) = " + f.format(mis) + "\n")
      log_file.write("## H(Selected | C) = " + f.format(ds.entropyPrefix - mis) + "\n")
      val denomH = ds.miEntireLabel + ds.entropyPrefix
      val denomG = ds.miEntireLabel * ds.entropyPrefix
      val muh = if(denomH == 0.0) 0.0 else 2*mis/denomH
      val mug = if(denomG == 0.0) 0.0 else mis/math.sqrt(denomG)
      log_file.write("## mu_H = " + f.format(muh) + "\n")
      log_file.write("## mu_G = " + f.format(mug) + "\n")

      log_file.write("\n# Events: function name:time (nsec):arguments\n")
      log_file.write("## sortAttrs:time:#_features\n")
      log_file.write("## sortCases:time:#_features\n")
      log_file.write("## findBorder:time:#_features\n")
        log_file.write(ds.events.map(e => e._1.name + ":" + e._2.mkString(":"))
          .mkString("", " ", "\n"))

      if(log == "high") {
        log_file.write("\n# Miscellany\n\n")
        log_file.write("## Entropy of features\n")
          log_file.write((0 to ds.maxAttr).map{i =>
            db.index2attr(i).name + ":" + f.format(ds.entropyAttr(i))}.mkString(" "))
        log_file.write("\n\n")
        log_file.write("## Entropy of features and labels\n")
          log_file.write((0 to ds.maxAttr).map{i =>
            db.index2attr(i).name + ":" + f.format(ds.entropyAttrLabel(i))}.mkString(" "))
        log_file.write("\n\n")
      }
      log_file.close()
      println("Log has been output to " + log_file_name + ".")

      // val statsAttr = (0 to ds.maxAttr).map{i =>
      //   val hf = ds.entropyAttr(i)
      //   val hfc = ds.entropyAttrLabel(i)
      //   val mi = hf + hc - hfc
      //   val su = 2*mi/(hf + hc)
      //   Vector(hf, hfc, mi, su)
      // }
      // log_file.write("## Entropy of features\n")
      // log_file.write((0 to ds.maxAttr).map{i => i + ":" + f.format(statsAttr(i)(0))}.mkString(" "))
      // log_file.write("\n\n")
      // log_file.write("## Entropy of features and labels\n")
      // log_file.write((0 to ds.maxAttr).map{i => i + ":" + f.format(statsAttr(i)(1))}.mkString(" "))
      // log_file.write("\n\n")
      // log_file.write("## Mutual information of features and labels\n")
      // log_file.write((0 to ds.maxAttr).map{i => i + ":" + f.format(statsAttr(i)(2))}.mkString(" "))
      // log_file.write("\n\n")
      // log_file.write("## Symmetical uncertainty of features and labels\n")
      // log_file.write((0 to ds.maxAttr).map{i => i + ":" + f.format(statsAttr(i)(3))}.mkString(" "))
      // log_file.write("\n\n")
    }
  }
}


