scalaVersion := "3.3.1"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

libraryDependencies += "nz.ac.waikato.cms.weka" % "weka-dev" % "3.7.12"

libraryDependencies += "com.github.scopt" %% "scopt" % "4.1.0"
libraryDependencies += "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"

resolvers += Resolver.sonatypeRepo("public")
javacOptions ++= Seq("-source", "11", "-target", "11")

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", _*) => MergeStrategy.discard
  case _                        => MergeStrategy.first
}

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "3.2.15" % Test,
  "org.scalatestplus" %% "mockito-4-6" % "3.2.15.0" % Test,
  "org.mockito" % "mockito-core" % "4.6.1" % Test,
  "org.scalameta" %% "munit" % "1.0.0" % Test
)