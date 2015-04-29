import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.Vectors

// Load and parse the data this is the output from the pyspark script
val data = sc.textFile("/Users/wgmueller/Desktop/dtm")
val parsedData = data.map(s => Vectors.dense(s.trim.split('\t').map(_.toDouble)))
// Index documents with unique IDs
val corpus = parsedData.zipWithIndex.map(_.swap).cache()

// Cluster the documents into three topics using LDA
val ldaModel = new LDA().setK(10).run(corpus)

// topics is an array of tuples (Long (document index), mllib.linalg.Vector (distribution over topics))
val topics = ldaModel.topicDistributions

// reading the raw data into an RDD
val raw = sc.textFile("/Users/wgmueller/Desktop/Data/output.csv/")

// create an RDD with index and id
val ids=raw.map(line =>line.split('\t')(0)).zipWithIndex.map(x => (x._2,x._1))

// maps (Long,mllib.linalg.Vector) to (Int,mllb.linalg.Vector) for the join
val topicsIdx = topics.map(x => (x._1.toInt,x._2))

// create schema for output format
case class topic_out = (id: String, dist:Array[Double])

// join ids and topic distributions convert to dataframe and then json string
val joined = ids.join(topicsIdx).map(x => topic_out(x._2._1,x._2._2.toArray)).toDF().toJSON

// save as text
joined.saveAsTextFile("/Users/wgmueller/Desktop/json_test")




