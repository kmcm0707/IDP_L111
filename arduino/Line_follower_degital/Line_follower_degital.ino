#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include <ArduinoMqttClient.h>
#include <WiFiNINA.h>
#include "utility/Adafruit_MS_PWMServoDriver.h"
#include <HCSR04.h>

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

String current_topic;

Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor *m1 = AFMS.getMotor(1); //left
Adafruit_DCMotor *m2 = AFMS.getMotor(2); //right    

// pins
int left_line_follower = 0;
int right_line_follower = 1;
int left_outer_follower = 2;
int right_outer_follower = 3;
int status_check = 0; //set to 0 at start set to 1 during line following set to 2 after picking up block

int speed;
String speed_str;

// defines pins numbers
const int trigPin = 9;
const int echoPin = 10;
long duration;
int distance;

UltraSonicDistanceSensor distancesensor(trigPin, echoPin);

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

  // topics can be unsubscribed using:
  // mqttClient.unsubscribe(topic);

  Serial.print("Topic: ");
  Serial.println(topic);
  Serial.print("Topic: ");
  Serial.println(topic2);

  Serial.println();
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin, INPUT); // Sets the echoPin as an Input

  pinMode(0, INPUT); 
  pinMode(1, INPUT);
  pinMode(2, INPUT); 
  pinMode(3, INPUT); 
}

void loop() {
  for(int c =0; c<10; c++){
    delay(100);
    mqttClient.poll();
  }
  // put your main code here, to run repeatedly:
  //servo_start();
  //start();
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
  int outerlines_passed = 0;
  while(true){
    mqttClient.poll();
    if(digitalRead(left_line_follower) == 1 &&  digitalRead(right_line_follower) == 0) // Turn left - right wheel faster
    {
      error = 5;
    }
    if(digitalRead(left_line_follower) == 0 &&  digitalRead(right_line_follower) == 1) // turn right - left wheel faster
    {
      error = -5;
    }
    if(digitalRead(left_line_follower) == 0 &&  digitalRead(right_line_follower) == 0) // ultrasound
    {
      if(status_check == 2) {
        double distance = distancesensor.measureDistanceCm();
        delay(50);
        error = (int)(distance - 5.85);
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
    if(digitalRead(left_line_follower) == 1  &&  digitalRead(right_line_follower) == 1) // both on line no error
    {
      error = 0;
      if(digitalRead(left_outer_follower) || digitalRead(right_outer_follower)){
        outerlines_passed = outerlines_passed + 1;
        if(outerlines_passed == 3){
          status_check = 2;
        }
      }
    }
    I = I + error;
    int D = error - last_error;
    last_error = error;

    int motorspeed = (int)(k_p*error + k_d * D + k_i * I);
    int leftspeed = basespeed - motorspeed;
    int rightspeed = basespeed + motorspeed;
    
    if(leftspeed < 0 || leftspeed > 255 || rightspeed < 0 || rightspeed > 255){
      if(leftspeed < 0){
        m1->setSpeed(0);
      } else {
        m1->run(FORWARD);
        m1->setSpeed(255);
      }
      if(rightspeed < 0){
        m2->setSpeed(0);
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
 
  Serial.println(speed);

  if (current_topic == topic){
    Serial.println("left");
    if(speed > 0){
      m1->run(FORWARD);
    } else{
      m1->run(BACKWARD);
      speed = -speed;
    }
    if(speed > 255){
      m1->setSpeed(255);
    } 
  }
  if (current_topic == topic2){
    Serial.println("right");
    if(speed > 0){
      m2->run(FORWARD);
    } else{
      m2->run(BACKWARD);
      speed = -speed;
    }
    if(speed > 255){
      m2->setSpeed(255);
    }
  }
  
  Serial.println();
  Serial.println();
}

void start(){
  m1->run(FORWARD);
  m2->run(FORWARD);
  m1->setSpeed(254);
  m2->setSpeed(210);
  delay(1000);
  while(!digitalRead(left_line_follower)  &&  !digitalRead(right_line_follower)){
    int i = 0;
  }
  status_check = 1;
  line_follower();
  
}

void servo_start(){
  
}
