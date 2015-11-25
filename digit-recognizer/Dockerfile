FROM aether/ubuntu-spark:14.04-1.5.1-2.7.1

ADD target/digit-recognizer.jar /digit-recognizer.jar

ENTRYPOINT ["/bin/sh", "-c", "$SPARK_HOME/bin/spark-submit --class com.kmalik.kaggle.digitrecognizer.Run digit-recognizer.jar $*"]
