#include <ArduinoJson.h>
#include <SPI.h>
#include <WiFiNINA.h>
#include "arduino_secrets.h"
#include <Adafruit_MotorShield.h>
#include <Wire.h>

///////please enter your sensitive data in the Secret tab/arduino_secrets.h
char ssid[] = "Diwakar's MacBook Pro";        // your network SSID (name)
char pass[] = "jyugemujyugemu";    // your network password (use for WPA, or use as key for WEP)


Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor *m1 = AFMS.getMotor(1); //left
Adafruit_DCMotor *m2 = AFMS.getMotor(2); //right 

int led =  LED_BUILTIN;
int status = WL_IDLE_STATUS;
int keyIndex = 0;            // your network key Index number (needed only for WEP)
// if you don't want to use DNS (and reduce your sketch size)

// use the numeric IP instead of the name for the server:

IPAddress server(128,232,255,41); 

//char server[] = "10.248.155.126";    // name address for Google (using DNS)
WiFiClient client;

const unsigned long wait_interval = 100L;
unsigned long last_connected = 0;

void printWifiStatus() {
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());
  // print your board's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);
  // print the received signal strength:
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}

void httpRequest(){
  client.stop();

  if (client.connect(server, 5000)) {
    Serial.println("connected to server");
    // Make a HTTP request:
    client.println("GET http://128.232.255.41:5000/data");
    client.println("Host: localhost");
    client.println("Connection: close");
    client.println();
  } else{
    Serial.println("connection failed");
  }
  last_connected = millis();
}

void setup() {
  //Initialize serial and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  if (!AFMS.begin()){
    while(1);
  }
  // check for the WiFi module:
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    // don't continue
    while (true);
  }
  
  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.println("Please upgrade the firmware");
  }
  // attempt to connect to Wifi network:
  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);
    // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
    status = WiFi.begin(ssid, pass);
    // wait 10 seconds for connection:
    delay(10000);
    WiFi.noLowPowerMode();
  }
  

  Serial.println("Connected to wifi");
  printWifiStatus();
  Serial.println("\nStarting connection to server...");
  // if you get a connection, report back via serial:
  httpRequest();
}

// this method makes a HTTP connection to the server:
/*
void httpRequest() {
  // close any connection before send a new request.
  // This will free the socket on the Nina module
  client.stop();
  
  // if there's a successful connection:
  if (client.connect(server, 80)) {
    Serial.println("connecting...");
    // send the HTTP PUT request:
    client.println("GET / HTTP/1.1");
    client.println("Host: example.org");
    client.println("User-Agent: ArduinoWiFi/1.1");
    client.println("Connection: close");
    client.println();
    // note the time that the connection was made:
    lastConnectionTime = millis();
  } else {
    // if you couldn't make a connection:
    Serial.println("connection failed");
  }
}
*/

void loop() {
  // if there are incoming bytes available
  // from the server, read them and print them:
  while (client.available()) {
    String payload = client.readString();
    Serial.println(payload);
    StaticJsonDocument<48> doc;
    DeserializationError error = deserializeJson(doc, payload);
    
    if (error) {
      Serial.print(F("deserializeJson() failed: "));
      Serial.println(error.f_str());
      return;
    }
    
    int leftspeed = (int)doc["llf"]; // 432
    int rightspeed = (int)doc["rlf"]; // -333
    Serial.print(leftspeed);
    Serial.print(" ,");
    Serial.println(rightspeed);

    if(leftspeed > 0 && leftspeed < 255) {
      Serial.println("set");
      m1->run(FORWARD);
      Serial.println("set");
      m1->setSpeed(leftspeed);
      Serial.println("set");
    }
    Serial.println("set");
    if(rightspeed > 0 && rightspeed < 255){
      Serial.println("set");
      m2->run(FORWARD);
      Serial.println("set");
      m2->setSpeed(rightspeed);
    }
    Serial.println("after if");
    
  } 
    
  
  if (millis() - last_connected > wait_interval){
    httpRequest();
  }

  // if the server's disconnected, stop the client:
  if (!client.connected()) {
    Serial.println();
    Serial.println("disconnecting from server.");
    client.stop();
    // do nothing forevermore:
    while (true);
  }
}
