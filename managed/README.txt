-Python3 wird benötigt um die Scripte auszuführen
- Mit "python visualizeGTA.py --plot 1" können die Daten visualisiert werden
- Die data_boxes.json hat folgenden Aufbau:

[
  {
    "$id": "1",                                   --> ignorieren
    "Detections": [                               --> Liste mit allen erfassten Fahrzeugen oder Menschen; jedes Objekt ist als Dictionary angelegt
      {
        "FURGame": {                              --> FUR == Front Upper Right Ecke der 3D Boundingbox; Game == in Spielkoordinaten
          "X": -244.065979,                       --> X, Y, Z in Meter innerhalb des Spiels vom Ursprung (0, 0, 0) entfernt
          "Y": -310.486877,
          "Z": 30.9894867
        },
        "FULGame": {
          "X": -243.59697,
          "Y": -309.361176,
          "Z": 30.9894867
        },
        "BULGame": {
          "X": -243.135422,
          "Y": -309.553467,
          "Z": 30.9894867
        },
        "BURGame": {
          "X": -243.604431,
          "Y": -310.679169,
          "Z": 30.9894867
        },
        "FLRGame": {
          "X": -244.065979,
          "Y": -310.486877,
          "Z": 28.8309956
        },
        "FLLGame": {
          "X": -243.59697,
          "Y": -309.361176,
          "Z": 28.8309956
        },
        "BLLGame": {
          "X": -243.135422,
          "Y": -309.553467,
          "Z": 28.8309956
        },
        "BLRGame": {
          "X": -243.604431,
          "Y": -310.679169,
          "Z": 28.8309956
        },
        "Type": "person",                           --> Typ des Objekts, person, car, bicycle, background usw.
        "cls": "Unknown",                           --> Objektklasse, bei person steht meist unknown da, bei car steht hier Autotyp, also z.B. SUV, Coupe, etc
        "Pos": {                                    --> Position des Objekts in 3D Spielkoordinaten
          "X": -243.600784,
          "Y": -310.0204,
          "Z": 30.1309948
        },
        "Rot": {                                    --> Rotation des Objekts in Spielangaben
          "X": 0.0,
          "Y": 0.0,
          "Z": 67.3817749
        },
        "Dim": {                                    --> Dimension des Objekts; X == 1,2m in die Breite, Y == 0.5m in die Tiefe, Z == 2,15m in die Höhe
          "X": 1.21949863,
          "Y": 0.50000006,
          "Z": 2.15849113
        },
        "Distance": 122.194252,                     --> Entfernung zwischen Objekt und der Aufnahmekamera
        "Speed": 0.0,                               --> Geschwindigkeit des Objekts in km/h zum Aufnahmezeitpunkt; nur wenn es ein Auto ist
        "Wheel": 0.0,                               --> Winkel der Räder zum Aufnahmezeitpunkt; nur wenn es ein Auto ist
        "Visibility": false,                        --> Gibt an, ob das Objekt sichtbar zur Kamera ist; bei "false" bedeutet dies, dass es im Bild verdeckt oder nicht sichtbar ist
        "DistCat": 9999,                            --> Falls Visibility == false, dann ist hier immer 9999; ansonsten wird hier die Entfernungskategorie angegeben: 15 == distance < 15m, 50 == distance < 50m usw.
        "NumberPlate": null,                        --> Kennzeichen von Fahrzeugen
        "Handle": 110659,                           --> das müsste eine eindeutige ID des Objekts sein
        "FUR": {                                    --> Front Upper Right Ecke der 3D Boundingbox in Bildkoordinaten; die Bildkoordinaten werden nur dann erzeugt, wenn "Visibility" auf true ist
          "X": 868,
          "Y": 138
        },
        "FUL": {
          "X": 878,
          "Y": 140
        },
        "BUL": {
          "X": 870,
          "Y": 140
        },
        "BUR": {
          "X": 861,
          "Y": 138
        },
        "FLL": {
          "X": 874,
          "Y": 175
        },
        "BLL": {
          "X": 866,
          "Y": 175
        },
        "BLR": {
          "X": 858,
          "Y": 172
        },
        "FLR": {
          "X": 865,
          "Y": 172
        },
        "BBmin": {                                  --> Obere linke Ecke der 2D Boundingbox in Bildkoordinaten, welche die 3D Boundingbox umpsannt; die Bildkoordinaten werden nur dann erzeugt, wenn "Visibility" auf true ist
          "X": 858,
          "Y": 138
        },
        "BBmax": {                                  --> Untere rechte Ecke der 2D Boundingbox in Bildkoordinaten, welche die 3D Boundingbox umpsannt; die Bildkoordinaten werden nur dann erzeugt, wenn "Visibility" auf true ist
          "X": 878,
          "Y": 175
        },
        "Pos2D": {                                  --> Mittelpunkt der 2D Boundingbox in Bildkoordinaten, welche die 3D Boundingbox umpsannt; die Bildkoordinaten werden nur dann erzeugt, wenn "Visibility" auf true ist
          "X": 868,
          "Y": 156
        },
      .....
    ],
    "Image": "gtav_cid0_c1794_1.tiff",              --> Bildname: cid0 == KreuzungsID 0 usw, c1794 == KameraID 1794 usw., die letzte Zahl im Namen gibt die Bildnummer an
    "RealTime": "2018-02-08T17:15:21.2441691Z",     --> Aufnahmezeitpunkt so wie es am Computer angezeigt wird
    "GameTime": "13:00:00",                         --> Uhrzeit im Spiel
    "ImageWidth": 1280,                             --> Bildbreite
    "ImageHeight": 720,                             --> Bildhöhe
    "UIwidth": 1280,                                --> Spielinterne Darstellungsbreite
    "UIheight": 720,                                --> Spielinterne Darstellungshöhe
    "CamHash": 1794,                                --> KameraID
    "GameTime2": 169472,                            --> vergangene Spielzeit in Millisekunden?! Leider gibt es keine weiteren Informationen hierüber
    "FrameTime": 0.01572089,                        --> vergangene Zeit um das Frame darzustellen?! Leider gibt es keine weiteren Informationen hierüber
    "CamFOV": 50.0,                                 --> Field of View der Kamera
    "CamNearClip": 0.15000000596046448,             --> Focallength
    "CamFarClip": 800.0,
    "Campos": {                                     --> Kameraposition in Spielkoordinaten
      "X": -198.309235,
      "Y": -422.916321,
      "Z": 40.7335472
    },
    ,
    "Camrot": {                                     --> Kamerarotation in Spielangaben
      "X": -22.401392,
      "Y": 0.0,
      "Z": -126.257256
    },
    "Camdir": {                                     --> Kamerarichtung
      "X": 0.7455187,
      "Y": -0.546781659,
      "Z": -0.381092817
    },
    "Carpos": {                                     --> Autoposition des Spielers, falls er im Auto sitzt
      "X": 0.0,
      "Y": 0.0,
      "Z": 0.0
    },
    "Carrot": {                                     --> Autorotation des Spielers, falls er im Auto sitzt
      "X": 0.0,
      "Y": 0.0,
      "Z": 0.0
    },
    "PMatrix": {                                    --> Projectionmatrix aus dem Speicher geladen
      "Values": [
        1.2062850616228806,
        2.6569864954245804e-09,
        1.1496775836943472e-13,
        -2.667636237621287e-17,
        -5.217909493371309e-09,
        2.144506974474929,
        2.1259171723367332e-13,
        0.0,
        1.265872395794787e-08,
        3.27436435743067e-08,
        1.5020370452414525e-05,
        -1.0,
        4.659801078332748e-07,
        1.1132193513674338e-06,
        0.15000225326493935,
        0.0
      ],
      "Storage": {
        "Data": [
          1.2062850616228806,
          2.6569864954245804e-09,
          1.1496775836943472e-13,
          -2.667636237621287e-17,
          -5.217909493371309e-09,
          2.144506974474929,
          2.1259171723367332e-13,
          0.0,
          1.265872395794787e-08,
          3.27436435743067e-08,
          1.5020370452414525e-05,
          -1.0,
          4.659801078332748e-07,
          1.1132193513674338e-06,
          0.15000225326493935,
          0.0
        ],
        "RowCount": 4,
        "ColumnCount": 4
      },
      "ColumnCount": 4,
      "RowCount": 4
    },
    "VMatrix": {                                      --> Viewmatrix aus dem Speicher geladen
      "Values": [
        -0.591411902062237,
        0.3073017132974141,
        -0.7455184031680981,
        0.0,
        -0.8063696369178323,
        -0.2253828309216422,
        0.5467820241753408,
        0.0,
        0.0,
        0.9245367646217346,
        0.3810928463935852,
        0.0,
        -458.30931978248526,
        -72.03697237244421,
        67.8765953978542,
        1.0
      ],
      "Storage": {
        "Data": [
          -0.591411902062237,
          0.3073017132974141,
          -0.7455184031680981,
          0.0,
          -0.8063696369178323,
          -0.2253828309216422,
          0.5467820241753408,
          0.0,
          0.0,
          0.9245367646217346,
          0.3810928463935852,
          0.0,
          -458.30931978248526,
          -72.03697237244421,
          67.8765953978542,
          1.0
        ],
        "RowCount": 4,
        "ColumnCount": 4
      },
      "ColumnCount": 4,
      "RowCount": 4
    },
    "Gamerpos": {                                       --> Position des Spielers selbst
      "X": -198.309235,
      "Y": -422.916321,
      "Z": 41.7335472
    }
  },
  ....
]