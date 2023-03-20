/*/////////////////////////////////////////////////////////////////////////////////////////////////
 * 
 *                              Skillz Makers Drones Competition
 *                              Hakfar Hayarok's UAV program
 *                              Group: Adar Slapak, Alon fueredi, Itamar Lin
 *                              
 *////////////////////////////////////////////////////////////////////////////////////////////////


#include <SoftwareSerial.h>
SoftwareSerial mySerial(3, 4);

char input_value = '0'; //the signal value

int trigPin = A0;  // TRIG pin
int echoPin = A1;//pin

float duration_us, distance_cm;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  mySerial.begin(9600);

  pinMode(trigPin, OUTPUT);
  // configure the echo pin to input mode
  pinMode(echoPin, INPUT);
}

void loop() {
  // put your main code here, to run repeatedly:

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // measure duration of pulse from ECHO pin
  duration_us = pulseIn(echoPin, HIGH);

  // calculate the distance
  distance_cm = 0.017 * duration_us
  delay(500);

  if (mySerial.available() > 0)//if getting a signal
  {
    //for debugging the distance 
    Serial.print("distance: ");
    Serial.print(distance_cm);
    Serial.println(" cm");
    
    //for debugging the signal
    input_value = mySerial.read();
    Serial.print(input_value);
    Serial.print("\n");
    
    mySerial.print(distance_cm);//sending the distance value
  }
}
