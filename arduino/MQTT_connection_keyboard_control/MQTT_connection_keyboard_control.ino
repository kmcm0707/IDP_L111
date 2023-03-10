/*
This was the final code used in the final robot.
It implements the servo, ultrasound (unused in the end) and motor control all from the MQTT connection.
*/
#include <ArduinoMqttClient.h>
#include <WiFiNINA.h>
#include <Adafruit_MotorShield.h>
#include <Servo.h>
#include <HCSR04.h>
// #include "arduino_secrets.h"

///////please enter your sensitive data in the Secret tab/arduino_secrets.h
// char ssid[] = "qwertyuiop";        // your network SSID (name)
// char pass[] = "asdfghjkl";     // your network password
// char ssid[] = "DevPhone";        // your network SSID (name)
// char pass[] = "jyugemujyugemu";     // your network password
char ssid[] = "DESKTOP-E1TS9EK_1488";        // your network SSID (name)
char pass[] = "46X)i457";  

WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);

const char broker[] = "test.mosquitto.org";
int        port     = 1883;

//topics to control the robot
String topic  = "IDP_2023_Follower_left_speed";
String topic2  = "IDP_2023_Follower_right_speed";
String vert_servo = "IDP_2023_Servo_Vertical";
String hori_servo = "IDP_2023_Servo_Horizontal";
String set_Ultra = "IDP_2023_Set_Ultrasound";
String set_BlockCheck = "IDP_2023_Set_Block";

//pins
const int trigPin = 6;
const int echoPin = 7;
int servo_vertical_pin = 9;
int servo_horizontal_pin = 10;
int motor_LED_pin = 2;
int colour_pin = 1;
int switch_pin = 0;

//servo angles
int vertical_angle_high = 160; //120?
int vertical_angle_low = 55;
int horizontal_angle_high =  110; //95 -- CLOSED
int horizontal_angle_low = 0; // -- OPEN
int vertical_angle_middle = 150;
int drop_block_angle = horizontal_angle_high - 20;

//ultrasound
UltraSonicDistanceSensor distancesensor(trigPin, echoPin);
bool enable_Ultrasound = false;

Servo servo_vertical;
Servo servo_horizontal;

String current_topic;

Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor *m1 = AFMS.getMotor(1); //left
Adafruit_DCMotor *m2 = AFMS.getMotor(2); //right   

int speed;
String speed_str;

bool checkBlock = false;

void move_servo_new(Servo servo, int pin_no, int start, int end){
  // Mover servo from start position to end position
  servo.attach(pin_no);
  if (start < end) {
    // From arduino 'servo' example:
    for (int pos = start; pos <= end; pos += 1) {
      // in steps of 1 degree
      servo.write(pos);              // tell servo to go to position in variable 'pos'
      delay(15);
    }
  } else {
    for (int pos = start; pos >= end; pos -= 1) {
      servo.write(pos);
      delay(15);
    }
  servo.detach();
  }
}

void setup() {
  //Initialize serial and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  if (!AFMS.begin()) {
    while (1);
  }

  servo_horizontal.attach(servo_horizontal_pin);
  servo_vertical.attach(servo_vertical_pin);

  servo_vertical.write(vertical_angle_high);

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

  // subscribe to a topic
  mqttClient.subscribe(topic);
  mqttClient.subscribe(topic2);
  mqttClient.subscribe(vert_servo);
  mqttClient.subscribe(hori_servo);
  mqttClient.subscribe(set_Ultra);
  mqttClient.subscribe(set_BlockCheck);

  // topics can be unsubscribed using:
  // mqttClient.unsubscribe(topic);
  pinMode(motor_LED_pin, OUTPUT);
  pinMode(colour_pin, INPUT);
  pinMode(switch_pin, INPUT);
  Serial.println();
}

void loop() {
  // call poll() regularly to allow the library to receive MQTT messages and
  // send MQTT keep alive which avoids being disconnected by the broker
  // ultra sound PID controller
  int error = 0; //error - turn left is +ve ,  turn right is -ve
  int last_error = 0;
  int I = 0;
  int basespeed = 180;
  const float k_i = 0.0001;
  const float k_p = 30;
  const float k_d = 10;
  mqttClient.poll();
  while(distancesensor.measureDistanceCm() < 30 && enable_Ultrasound){
    // ultra sound PID controller
    mqttClient.poll();
    error = -(7.53 - distancesensor.measureDistanceCm()); 
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
      } else{
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
  while(checkBlock == true){
    // check if block is red or blue
    if(!digitalRead(switch_pin)){
      if(digitalRead(colour_pin)){
        mqttClient.beginMessage("IDP_2023_Color");
        mqttClient.print("0"); //red
        mqttClient.endMessage();
        checkBlock = false;
      } else {
        mqttClient.beginMessage("IDP_2023_Color");
        mqttClient.print("1"); //blue or nothing
        mqttClient.endMessage();
        checkBlock = false;
      }
    } else {
      // if switch is not pressed, open claw to drop block and then pick up again (to change angle of block)
      move_servo_new(servo_vertical, servo_vertical_pin, vertical_angle_high, vertical_angle_low);
      move_servo_new(servo_horizontal, servo_horizontal_pin, horizontal_angle_high, horizontal_angle_low);
      move_servo_new(servo_horizontal, servo_horizontal_pin, horizontal_angle_low, horizontal_angle_high);
      move_servo_new(servo_vertical, servo_vertical_pin, vertical_angle_low, vertical_angle_middle);
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

  // CONTROL MOTORS
  if (current_topic == topic){
    digitalWrite(motor_LED_pin, HIGH);
    Serial.println("left");
    if(speed > 0){
      m1->run(FORWARD);
    } else{
      m1->run(BACKWARD);
      speed = -speed;
    }
    if(speed > 255){
      m1->setSpeed(255);
    } else {
      m1->setSpeed(speed);
    }
    if(speed == 0){
      digitalWrite(motor_LED_pin, LOW);
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
    } else {
      m2->setSpeed(speed);
    }
    
  }

  if (current_topic == vert_servo){
    digitalWrite(motor_LED_pin, LOW);
    if (speed == 1) {
      // RAISE CLAW
      move_servo_new(servo_vertical, servo_vertical_pin, vertical_angle_low, vertical_angle_high);
    } else if (speed == 0) {
      // LOWER CLAW
      move_servo_new(servo_vertical, servo_vertical_pin, vertical_angle_high, vertical_angle_low);
    } else if (speed == 2) {
      // RAISE CLAW HALFWAY
      move_servo_new(servo_vertical, servo_vertical_pin, vertical_angle_low, vertical_angle_middle);
    } else if (speed == 3) {
      // RAISE CLAW FROM HALF TO FULL HEIGHT
      move_servo_new(servo_vertical, servo_vertical_pin, vertical_angle_middle, vertical_angle_high);
    }
  }

  if (current_topic == hori_servo){
    digitalWrite(motor_LED_pin, LOW);
    if (speed == 1){
      // CLOSE CLAW (low -> high)
      move_servo_new(servo_horizontal, servo_horizontal_pin, horizontal_angle_low, horizontal_angle_high);
    } else if (speed == 0){
      // OPEN CLAW (high -> low)
      move_servo_new(servo_horizontal, servo_horizontal_pin, horizontal_angle_high, horizontal_angle_low);
    } else if (speed == 2){
      // DROP BLOCK
      move_servo_new(servo_horizontal, servo_horizontal_pin, horizontal_angle_high, drop_block_angle);
      delay(500);
      move_servo_new(servo_horizontal, servo_horizontal_pin, drop_block_angle, horizontal_angle_high);
    }
  }

  /*

  if (current_topic == vert_servo){
     servo_vertical.attach(servo_vertical_pin);
    if (speed == 1){
      // From arduino 'servo' example:
      for (int pos = vertical_angle_low; pos <= vertical_angle_high; pos += 1) {
      // in steps of 1 degree
      servo_vertical.write(pos);              // tell servo to go to position in variable 'pos'
      delay(15);
      }                             // waits 15 ms for the servo to reach the position
    } else{
      // From arduino 'servo' example:
      for (int pos = vertical_angle_high; pos >= vertical_angle_low; pos -= 1) {
      // in steps of 1 degree
      servo_vertical.write(pos);              // tell servo to go to position in variable 'pos'
      delay(15);                                // waits 15 ms for the servo to reach the position
      }
    }
    servo_vertical.detach();
  }

  if (current_topic == hori_servo){
    servo_vertical.attach(servo_horizontal_pin);
    if(speed == 1){
      for (int pos = horizontal_angle_low; pos <= horizontal_angle_high; pos += 1) {
      // in steps of 1 degree
      servo_horizontal.write(pos);              // tell servo to go to position in variable 'pos'
      delay(15);                                // waits 15 ms for the servo to reach the position

      }
    } else {
      for (int pos = horizontal_angle_high; pos >= horizontal_angle_low; pos -= 1) {
      // in steps of 1 degree
      servo_horizontal.write(pos);              // tell servo to go to position in variable 'pos'
      delay(15);                                // waits 15 ms for the servo to reach the position
      }
    }
  }

  */

  // ULTRASOUND Enable
  if (current_topic == set_Ultra){
    if(speed == 1){
      enable_Ultrasound = true;
    } else {
      enable_Ultrasound = false;
    }
  }

  // BLOCK CHECK
  if (current_topic == set_BlockCheck){
    if(speed == 1){
      checkBlock = true;
    } else {
      checkBlock = false;
    }
  }
  
  Serial.println();
  Serial.println();
}
