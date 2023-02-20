// Code that was used to test the ultrasound sensor
#include <HCSR04.h>
const int trigPin = 6;
const int echoPin = 7;
UltraSonicDistanceSensor distancesensor(trigPin, echoPin);

void setup() {
  Serial.begin(9600); // Starts the serial communication
}

void loop() {
  Serial.println(distancesensor.measureDistanceCm());
  delay(100);
  
}
// 7.53 = correct
// 4.5 - smallest allowed
// 10.41 - larget allowed
