#include <ArduinoMqttClient.h>
#include <WiFiNINA.h>
#include <Adafruit_MotorShield.h>
// #include "arduino_secrets.h"

///////please enter your sensitive data in the Secret tab/arduino_secrets.h
<<<<<<< HEAD
char ssid[] = "DESKTOP-E1TS9EK_1488";        // your network SSID (name)
char pass[] = "46X)i457";     // your network password
=======
char ssid[] = "DevPhone";        // your network SSID (name)
char pass[] = "jyugemujyugemu";     // your network password
>>>>>>> e41d5c7192859a3fb892ecf62037239417aed96f

WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);

const char broker[] = "test.mosquitto.org";
int        port     = 1883;
String topic  = "IDP_2023_Follower_left_speed";
String topic2  = "IDP_2023_Follower_right_speed";
//const char topic3[]  = "real_unique_topic_3";

String current_topic;

Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor *m1 = AFMS.getMotor(1); //left
Adafruit_DCMotor *m2 = AFMS.getMotor(2); //right   

int speed;
String speed_str;

void setup() {
  //Initialize serial and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  if (!AFMS.begin()) {
    while (1);
  }
  // attempt to connect to Wifi network:
  Serial.print("Attempting to connect to SSID: ");
  Serial.println(ssid);
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {
    // failed, retry
    Serial.print(".");
    delay(5000);
  }

  // turn on motor
  m1 ->run(RELEASE);
  m2 ->run(RELEASE);


  Serial.println("You're connected to the network");
  Serial.println();

  Serial.print("Attempting to connect to the MQTT broker: ");
  Serial.println(broker);

  if (!mqttClient.connect(broker, port)) {
    Serial.print("MQTT connection failed! Error code = ");
    Serial.println(mqttClient.connectError());

    while (1);
  }

  Serial.println("You're connected to the MQTT broker!");
  Serial.println();

  // set the message receive callback
  mqttClient.onMessage(onMqttMessage);

  Serial.print("Subscribing to topic: ");
  Serial.println(topic);
  Serial.println();

  // subscribe to a topic
  mqttClient.subscribe(topic);
  mqttClient.subscribe(topic2);
  // mqttClient.subscribe(topic3);

  // topics can be unsubscribed using:
  // mqttClient.unsubscribe(topic);

  Serial.print("Topic: ");
  Serial.println(topic);
  Serial.print("Topic: ");
  Serial.println(topic2);
  // Serial.print("Topic: ");
  // Serial.println(topic3);

  Serial.println();
}

void loop() {
  // call poll() regularly to allow the library to receive MQTT messages and
  // send MQTT keep alive which avoids being disconnected by the broker
  /*m1 ->setSpeed(150);
  m1 ->run(FORWARD);
  m2 ->setSpeed(150);
  m2 ->run(FORWARD);*/

  mqttClient.poll();
}

void onMqttMessage(int messageSize) {
  // we received a message, print out the topic and contents
  Serial.println("Received a message with topic '");
  current_topic = mqttClient.messageTopic();
  Serial.print(current_topic);
  Serial.print("', length ");
  Serial.print(messageSize);
  Serial.println(" bytes:");

  // use the Stream interface to print the contents
  speed_str = mqttClient.readString();
  speed = speed_str.toInt();
  /*
  if(messageSize == 1){
  } else{
    while (mqttClient.available()) {
      for(int i=0; i<messageSize; i++){
        speed_str[i] = mqttClient.read();
      }
    
    speed_str[messageSize] = "\0";
    speed = (int)speed_str;

  }*/
  Serial.println(speed);

  if (current_topic == topic){
    Serial.println("left");
    if(speed > 0){
      m1->run(FORWARD);
    } else{
      m1->run(BACKWARD);
      speed = -speed;
    }
    m1->setSpeed(speed);
  }
  if (current_topic == topic2){
    Serial.println("right");
    if(speed > 0){
      m2->run(FORWARD);
    } else{
      m2->run(BACKWARD);
      speed = -speed;
    }
    m2->setSpeed(speed);
  }
  
  Serial.println();
  Serial.println();
}
