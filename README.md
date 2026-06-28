# Frontfläche

Eine installierbare PWA, die live die projizierte **Stirnfläche** eines Skifahrers
in m² schätzt – als Trainingshilfe, um die eigene Hocke-Position aerodynamisch zu
optimieren. Komplett offline-fähig, ohne Laufzeit-Abhängigkeiten (keine CDNs,
kein App-Store).

## Messprinzip

1. **Fahrer** = dunkle Silhouette vor hellem Schnee. Erkennung über eine
   Helligkeitsschwelle, anschliessend wird nur die **grösste zusammenhängende
   dunkle Fläche** behalten (Schatten, Zuschauer, Speckle fallen weg).
2. **Ball** mit bekanntem Durchmesser auf Fusshöhe daneben = metrischer Massstab.
   Sein Pixel-Durchmesser liefert `s = D_real / d_px` (Meter pro Pixel).
3. **Stirnfläche** `A = N_Fahrer · s²` in m².

Die Berechnung ist auflösungsunabhängig: Ändert man die Verarbeitungsauflösung,
skalieren Pixelzahl und Ball-Durchmesser gemeinsam, das Resultat bleibt gleich.
Eine tiefere Auflösung erkauft nur mehr Tempo bei etwas mehr Rauschen.

## Bedienung

1. «Kamera starten» (Frontkamera, damit der Fahrer die Anzeige selbst sieht).
2. Fahrer mittig ins Bild, heller Hintergrund. Ball auf Fusshöhe daneben legen.
3. **Ball einmal antippen** – das setzt die Massstabsfarbe (Neonball ideal).
4. Tuck variieren. «Referenz setzen» speichert die aktuelle Pose; die Anzeige
   zeigt danach die prozentuale Abweichung. Der Bestwert (kleinste Fläche) wird
   automatisch mitgeführt.

Im Einstellungspanel (⚙︎): Dunkelschwelle, **Bodenlinie** (blendet Ski und
Vordergrund unten aus), Ball-Durchmesser, Auflösung, Ton-Feedback,
Kamera-Wechsel.

## Grenzen (bewusst so)

- Gemessen wird die Fläche **A**, nicht das Widerstandsprodukt c<sub>w</sub>·A.
  A ist aber genau der Teil, den der Fahrer über die Körperhaltung steuert.
- Gedacht für den Vergleich **eigener** Posen, nicht zwischen Athleten.
- Ski-/Fusslage zwischen den Posen konstant halten – diese Anteile sind ein
  ~konstanter Bias, der sich im Relativvergleich weghebt.
- Fahrer zentriert halten (Randverzeichnung der Weitwinkel-Frontkamera).

## Lokal testen

Ein statischer Server genügt (Kamera braucht HTTPS **oder** localhost):

```bash
python3 -m http.server 8000
# dann http://localhost:8000 im Browser öffnen
```

## Deployment auf Vercel

Reines statisches Projekt – kein Build nötig.

1. Ordner in ein GitHub-Repo pushen.
2. In Vercel «New Project» → Repo importieren → Framework Preset **Other** →
   Deploy. Vercel liefert die Dateien direkt aus, HTTPS ist automatisch (nötig
   für `getUserMedia`).

Alternativ per CLI:

```bash
npm i -g vercel
vercel
```

## Dateien

| Datei | Zweck |
|-------|-------|
| `index.html` | Oberfläche und Live-Anzeige |
| `app.js` | Kamera, Masken, Connected-Components, Massstab, Flächenberechnung |
| `style.css` | Instrument-/HUD-Optik |
| `manifest.webmanifest` | PWA-Metadaten |
| `sw.js` | Service Worker (Offline, cache-first) |
| `icons/` | App-Icons |

## Mögliche Ausbaustufen

- Zweites Segmentierungs-Backend (MediaPipe Selfie/Multiclass) als Alternative
  für Bedingungen ohne sauberen Schnee-Kontrast.
- Tiefensensor (ARKit/ARCore LiDAR) für metrisch korrekten Massstab ganz ohne
  Ball und tiefenkorrigierte Flächenintegration (dann native App statt PWA).
- Aufzeichnung/Export der Posen samt Werten für die Trainingsdokumentation.
