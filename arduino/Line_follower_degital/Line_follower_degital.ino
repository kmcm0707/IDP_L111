#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include <ArduinoMqttClient.h>
#include <WiFiNINA.h>
#include "utility/Adafruit_MS_PWMServoDriver.h"

///////please enter your sensitive data in the Secret tab/arduino_secrets.h
char ssid[] = "DESKTOP-E1TS9EK_1488";        // your network SSID (name)
char pass[] = "46X)i457";     // your network password
//char ssid[] = "DevPhone";        // your network SSID (name)
//char pass[] = "jyugemujyugemu";     // your network password

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


int left_line_follower = 0;
int right_line_follower = 1;

int linefollower_trigger = 500;
int status_check = 0; //set to 1 after blocks picked up
void setup() {
  // put your setup code here, to run once:

   if (!AFMS.begin()) {
    while (1);
  }

  Serial.print("Attempting to connect to SSID: ");
  Serial.println(ssid);
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {
    // failed, retry
    Serial.print(".");
    delay(5000);
  }

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
  for(int c =0; c<10; c++){
    delay(100);
    mqttClient.poll();
  }
  // put your main code here, to run repeatedly:
  line_follower();
}

void line_follower(){
  m1->run(FORWARD);
  m2->run(FORWARD);
  int error = 0; //error - turn left is +ve ,  turn right is -ve
  int last_error = 0;
  int I = 0;
  int basespeed = 200;
  const float k_i = 0.001;
  const float k_p = 30;
  const float k_d = 10;
  while(true){
    
    if(digitalRead(left_line_follower) &&  ! digitalRead(right_line_follower)) // Turn left - right wheel faster
    {
      error = 5;
    }
    if(!digitalRead(left_line_follower) &&  digitalRead(right_line_follower)) // turn right - left wheel faster
    {
      error = -5;
    }
    if(!digitalRead(left_line_follower) <  &&  !digitalRead(right_line_follower)) // ultrasound
    {
      if(status_check == 1) {
        /*digitalWrite(trigPin, LOW);
        delayMicroseconds(2);
        // Sets the trigPin on HIGH state for 10 micro seconds
        digitalWrite(trigPin, HIGH);
        delayMicroseconds(10);
        digitalWrite(trigPin, LOW);
        // Reads the echoPin, returns the sound wave travel time in microseconds
        duration = pulseIn(echoPin, HIGH);
        // Calculating the distance
        distance = duration * 0.034 / 2;
        // Prints the distance IN CM
        //error = reference - distance;
        */
      }
      else {
        error = last_error;
        if(error < 0){
          error = -10;
        }
        if (error > 0){
          error = 10;
        }
      }
    }
    if(digitalRead(left_line_follower)  &&  digitalRead(right_line_follower)) // turn right - left wheel faster
    {
      error = 0;
    }
    I = I + error;
    int D = error - last_error;
    last_error = error;

    int motorspeed = (int)(k_p*error + k_d * D + k_i * I);
    int leftspeed = basespeed - motorspeed;
    int rightspeed = basespeed + motorspeed;
    
    if(leftspeed < 0 || leftspeed > 255 || rightspeed < 0 || rightspeed > 255){
      if(leftspeed < 0){
        m1->run(BACKWARD);
        if(leftspeed > -255){
          m1->setSpeed(-leftspeed);
        } else {
          m1->setSpeed(255);
        }
      } else {
        m1->run(FORWARD);
        m1->setSpeed(255);
      }
      if(rightspeed < 0){
        m2->run(BACKWARD);
        if(rightspeed > -255){
          m2->setSpeed(-rightspeed);
        } else {
          m2->setSpeed(255);
        }
      } else {
        m2->run(FORWARD);
        m2->setSpeed(255);
      }
    }
    if(leftspeed > 0 && leftspeed < 255) {
      m1->run(FORWARD);
      m1->setSpeed(leftspeed);
    }
    if(rightspeed > 0 && rightspeed < 255){
      m2->run(FORWARD);
      m2->setSpeed(rightspeed);
    }
    
  }
  
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
