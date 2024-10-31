
## Step 1: Get Kafka
Download  https://www.apache.org/dyn/closer.cgi?path=/kafka/3.8.1/kafka_2.13-3.8.1.tgz 
extract it
```shell
tar -xzf kafka_2.13-3.8.1.tgz
cd kafka_2.13-3.8.1
```

## Step 2: Start the Kafka environment


Kafka with KRaft
Kafka can be run using KRaft mode using local scripts and downloaded files or the docker image. Follow one of the sections below but not both to start the kafka server.

Using downloaded files
Generate a Cluster UUID

```shell
KAFKA_CLUSTER_ID="$(bin/kafka-storage.sh random-uuid)"
```
Format Log Directories

```shell
bin/kafka-storage.sh format -t $KAFKA_CLUSTER_ID -c config/kraft/server.properties
```
Start the Kafka Server

```shell
bin/kafka-server-start.sh config/kraft/server.properties
```

## Step 3: Create a topic to store your events

Kafka is a distributed event streaming platform that lets you read, write, store, and process events (also called records or messages in the documentation) across many machines.

So before you can write your first events, you must create a topic. Open another terminal session and run:

```shell
bin/kafka-topics.sh --create --topic quickstart-events --bootstrap-server localhost:9092
```

show you details such as the partition count of the new topic:
```shell
bin/kafka-topics.sh --describe --topic quickstart-events --bootstrap-server localhost:9092

# Topic: quickstart-events        TopicId: NPmZHyhbR9y00wMglMH2sg PartitionCount: 1       ReplicationFactor: 1	Configs:
# Topic: quickstart-events Partition: 0    Leader: 0   Replicas: 0 Isr: 0
```

## Step 4: Write some events into the topic

```shell
bin/kafka-console-producer.sh --topic quickstart-events --bootstrap-server localhost:9092
>This is my first event
>This is my second event
```
## Step 5: Read the events

```shell
bin/kafka-console-consumer.sh --topic quickstart-events --from-beginning --bootstrap-server localhost:9092
This is my first event
This is my second event
```

## Step 6: Import/export your data as streams of events with Kafka Connect

Edit the config/connect-standalone.properties file, add or change the plugin.path configuration property match the following, and save the file:

```shell
echo "plugin.path=libs/connect-file-3.8.1.jar" >> config/connect-standalone.properties
```

Then, start by creating some seed data to test with:

```shell
echo -e "foo\nbar" > test.txt
```

Next, we'll start two connectors running in standalone mode
```shell
bin/connect-standalone.sh config/connect-standalone.properties config/connect-file-source.properties config/connect-file-sink.properties

```
 We can verify the data has been delivered through the entire pipeline by examining the contents of the output file:
 
```shell
more test.sink.txt
# foo
# bar
```

Note that the data is being stored in the Kafka topic connect-test, so we can also run a console consumer to see the data in the topic (or use custom consumer code to process it):

```shell
bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic connect-test --from-beginning
#{"schema":{"type":"string","optional":false},"payload":"foo"}
#{"schema":{"type":"string","optional":false},"payload":"bar"}
```

The connectors continue to process data, so we can add data to the file and see it move through the pipeline:

```shell
echo "Another line" >> test.txt
```
