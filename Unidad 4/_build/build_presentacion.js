// Genera Actividad 3/Presentacion_A3_Grupo12_Iriarte_Trochez.pptx
// Uso: NODE_PATH=<carpeta node_modules con pptxgenjs> node _build/build_presentacion.js [ruta_apply_theme.js]
const path = require("path");
const pptxgen = require("pptxgenjs");

const RAIZ = path.resolve(__dirname, "..");
const FIG = (f) => path.join(RAIZ, "salidas", f);
const SALIDA = path.join(RAIZ, "Actividad 3", "Presentacion_A3_Grupo12_Iriarte_Trochez.pptx");

const THEME = {
  name: "Milan U4",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1D2735", lt1: "FFFFFF", dk2: "0B2559", lt2: "EEF3F8",
    accent1: "0B2559", accent2: "1D6F5F", accent3: "9A6510", accent4: "52728A",
    accent5: "D1495B", accent6: "173A7A", hlink: "175582", folHlink: "52728A",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13,333 x 7,5 in
pres.author = "David Santiago Iriarte Zamora y Jose Adrian Trochez Cuellar";
pres.company = "Grupo 12";
pres.title = "Pronóstico a 10 minutos de la actividad de Internet en Milán";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const W = 13.333, M = 0.6, CW = W - 2 * M;

// ---------- Layouts ----------
pres.defineSlideMaster({
  title: "PORTADA",
  background: { color: C.text2 },
  objects: [],
});
pres.defineSlideMaster({
  title: "CONTENIDO",
  background: { color: C.background1 },
  margin: [0.4, M, 0.6, M],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: CW, h: 0.85, fontFace: THEME.headFontFace,
      fontSize: 34, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Grupo 12 · Iriarte y Trochez · Unidad 4", options: { x: M, y: 7.02, w: 8, h: 0.3, fontSize: 10,
      color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
});
pres.defineSlideMaster({
  title: "CIERRE",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: CW, h: 0.85, fontFace: THEME.headFontFace,
      fontSize: 34, bold: true, color: C.background1, valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: "CADCFC", align: "right" },
});

// ---------- Ayudas ----------
function bullets(slide, items, opt) {
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    const base = { bullet: { indent: 16 }, paraSpaceAfter: opt.gap ?? 8 };
    if (Array.isArray(it)) {
      runs.push({ text: it[0] + " ", options: { ...base, bold: true, color: opt.labelColor || C.text2 } });
      runs.push({ text: it[1], options: { breakLine: !last } });
    } else {
      runs.push({ text: it, options: { ...base, breakLine: !last } });
    }
  });
  slide.addText(runs, { x: opt.x, y: opt.y, w: opt.w, h: opt.h, fontSize: opt.size || 16, color: opt.color || C.text1,
    valign: "top", margin: opt.margin ?? 0, isTextBox: true, fit: "none" });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 },
    line: { color: fill || C.background2, width: 0 } });
}

function stat(slide, x, y, w, h, big, label, color, fill) {
  card(slide, x, y, w, h, fill);
  slide.addText(big, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: h * 0.5, fontFace: THEME.headFontFace, fontSize: 30, bold: true,
    color: color || C.text2, valign: "bottom", margin: 0, isTextBox: true });
  slide.addText(label, { x: x + 0.2, y: y + 0.14 + h * 0.5, w: w - 0.4, h: h * 0.5 - 0.16, fontSize: 13, color: C.text1,
    valign: "top", margin: 0, isTextBox: true });
}

function numCircle(slide, x, y, n, fill, color) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color: fill || C.text2 }, line: { color: fill || C.text2, width: 0 } });
  slide.addText(String(n), { x, y, w: 0.42, h: 0.42, fontSize: 14, bold: true, color: color || C.background1, align: "center",
    valign: "middle", margin: 0, isTextBox: true });
}

function caption(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.32, fontSize: 11, italic: true, color: C.accent4, margin: 0, isTextBox: true });
}

const HDR = (t) => ({ text: t, options: { bold: true, color: C.text2, fill: { color: C.background2 } } });
const N = (t, extra) => ({ text: t, options: { align: "right", ...(extra || {}) } });
const BEST = { bold: true, fill: { color: "E3F1EC" } };
const tableOpts = (x, y, w, colW, size) => ({ x, y, w, colW, fontSize: size || 14, color: C.text1, fontFace: THEME.bodyFontFace,
  border: { type: "solid", pt: 0.75, color: "D5DCE4" }, margin: [3, 6, 3, 6], valign: "middle" });

// ---------- 1. Portada ----------
pres.addSection({ title: "Introducción" });
let s = pres.addSlide({ masterName: "PORTADA", sectionTitle: "Introducción" });
s.addText("TÉCNICAS AVANZADAS DE MODELADO EN IA · UNIDAD 4 · PROYECTO INTEGRADOR", { x: M, y: 1.55, w: CW, h: 0.4, fontSize: 13,
  charSpacing: 2, color: "9FB4D3", margin: 0, isTextBox: true });
s.addText("Pronóstico a 10 minutos de la actividad de Internet en Milán", { x: M, y: 2.05, w: 10.8, h: 1.7,
  fontFace: THEME.headFontFace, fontSize: 44, bold: true, color: C.background1, valign: "top", margin: 0, isTextBox: true });
s.addText("Redes recurrentes LSTM y GRU frente a líneas base sencillas, con la misma partición cronológica", { x: M, y: 3.85, w: 11.6, h: 0.8,
  fontSize: 20, italic: true, color: "CADCFC", margin: 0, isTextBox: true });
s.addText("David Santiago Iriarte Zamora y Jose Adrian Trochez Cuellar · Grupo 12", { x: M, y: 5.25, w: CW, h: 0.45, fontSize: 18,
  bold: true, color: C.background1, margin: 0, isTextBox: true });
s.addText("github.com/dsiriarte/Tecnicas-Avanzadas-de-Modelado-en-IA/tree/main/Unidad 4", { x: M, y: 5.75, w: CW, h: 0.4, fontSize: 13,
  color: "CADCFC", margin: 0, isTextBox: true,
  hyperlink: { url: "https://github.com/dsiriarte/Tecnicas-Avanzadas-de-Modelado-en-IA/tree/main/Unidad%204" } });

// ---------- 2. Problema ----------
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Introducción" });
s.addText("El reto: superar a la persistencia", { placeholder: "title" });
s.addText("¿Una red recurrente reduce el error de pronóstico frente a líneas base sencillas, evaluadas con la misma partición?",
  { x: M, y: 1.3, w: CW, h: 0.5, fontSize: 18, italic: true, color: C.accent4, margin: 0, isTextBox: true });
bullets(s, [
  ["Para qué sirve.", "Ajustar la capacidad de la red, encender o apagar recursos de radio y detectar fallas frente a lo previsto."],
  ["Por qué es difícil.", "El cambio mediano entre intervalos es 8.462, cerca del 1 % del nivel. Repetir el último valor ya da un MAPE de 1,6 %."],
  ["Hipótesis.", "La red puede aprovechar la inercia de la serie (autocorrelación del cambio de 0,67) y la hora del día."],
  ["Criterio de éxito.", "Menor MAE en prueba, con un IC 95 % que excluya el cero y Diebold-Mariano p < 0,05."],
], { x: M, y: 2.05, w: 7.4, h: 4.6, size: 17, gap: 14 });
stat(s, 8.55, 2.05, 4.18, 1.4, "39 millones", "registros de 10.000 celdas, del 4 al 10 de noviembre de 2013");
stat(s, 8.55, 3.65, 4.18, 1.4, "1.008", "intervalos de 10 minutos al agregar la ciudad");
stat(s, 8.55, 5.25, 4.18, 1.4, "9.906 vs. 9.230", "MAE en prueba del avance: la LSTM perdía contra la persistencia", C.accent3, "FBF3E4");

// ---------- 3. Datos ----------
pres.addSection({ title: "Método" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Método" });
s.addText("Datos y partición cronológica", { placeholder: "title" });
s.addImage({ path: FIG("fig1_serie_particion.png"), x: M, y: 1.35, w: 8.1, h: 8.1 / 2.308, altText: "Serie agregada y registros válidos con la partición" });
caption(s, "Amarillo: validación (sábado). Rojo: prueba (domingo). Hora local de Milán.", M, 1.35 + 8.1 / 2.308 + 0.02, 8.1);
bullets(s, [
  ["Fuente.", "Telecom Italia Big Data Challenge (Barlacchi et al., 2015), licencia ODbL."],
  ["Agregación.", "Suma de internet por intervalo; no reemplazamos por cero el 49,7 % de valores vacíos."],
  ["Sin fuga.", "Escaladores ajustados solo con entrenamiento; un assert verifica que ningún objetivo cruce particiones."],
], { x: 9.0, y: 1.4, w: 3.73, h: 3.9, size: 15, gap: 12 });
stat(s, M, 5.35, 3.85, 1.35, "693 / 151 / 152", "ventanas: lunes a viernes, sábado y domingo");
stat(s, M + 4.12, 5.35, 3.85, 1.35, "771 y 642 mil", "nivel medio en días laborales y el domingo");
stat(s, M + 8.24, 5.35, 3.86, 1.35, "r = 0,88", "correlación entre la actividad y la cobertura");

// ---------- 4. Pipeline y arquitectura ----------
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Método" });
s.addText("Pipeline y arquitectura de la red", { placeholder: "title" });
const pasos = [
  ["Carga verificada", "SHA-256 del CSV, 1.008 intervalos"],
  ["Ventana 12 × 4", "nivel, cambio, seno y coseno de la hora"],
  ["GRU(32)", "3.648 parámetros"],
  ["Dropout 0,4", "regulariza el estado final"],
  ["Dense(1)", "cambio escalado"],
  ["Pronóstico", "último valor más el cambio; promedio de 3 semillas"],
];
const bw = (CW - 5 * 0.22) / 6;
pasos.forEach((p, i) => {
  const x = M + i * (bw + 0.22);
  card(s, x, 1.55, bw, 1.45);
  numCircle(s, x + 0.12, 1.38, i + 1);
  s.addText([{ text: p[0], options: { bold: true, color: C.text2, breakLine: true, fontSize: 15 } }, { text: p[1], options: { fontSize: 14 } }],
    { x: x + 0.12, y: 1.82, w: bw - 0.24, h: 1.13, color: C.text1, valign: "top", margin: 0, isTextBox: true });
});
s.addTable([
  [HDR("Decisión"), HDR("Evidencia")],
  ["Predecir el cambio", "El cambio es 15 veces más pequeño que el nivel; con este objetivo, la persistencia equivale a predecir cero."],
  ["GRU y no LSTM", "Gana en las 4 combinaciones de la ronda 2, con 23 % menos parámetros (3.681 frente a 4.769)."],
  ["32 unidades", "Con 693 ventanas de entrenamiento, el modelo tiene unos 5 parámetros por ejemplo."],
  ["Hora en seno y coseno", "23:50 y 00:00 quedan cerca; con objetivo de cambio, el MAE pasa de 6.366 a 5.844."],
  ["MSE y MAE", "MSE para entrenar; MAE para reportar, en las mismas unidades de la serie."],
], tableOpts(M, 3.45, CW, [3.2, CW - 3.2], 17));

// ---------- 5. Diseño experimental ----------
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Método" });
s.addText("Diseño experimental: 147 redes", { placeholder: "title" });
s.addTable([
  [HDR("Ronda"), HDR("Qué cambia"), { text: "Mejor MAE val.", options: { bold: true, color: C.text2, fill: { color: C.background2 }, align: "right" } },
    { text: "vs. persistencia", options: { bold: true, color: C.text2, fill: { color: C.background2 }, align: "right" } }],
  ["Líneas base", "persistencia · deriva · Ridge", N("9.147 · 7.612 · 5.875"), N("—")],
  ["R1 · avance", "un factor por vez desde E1: lr, dropout, celda u optimizador", N("6.727 (E4 GRU)"), N("−26,5 %")],
  ["R2 · factorial 2 × 2 × 2", "celda × objetivo × entradas", N("5.844"), N("−36,1 %")],
  [{ text: "R3 · malla de 36", options: BEST }, { text: "lr × dropout × optimizador × celda", options: BEST }, N("5.653", BEST), N("−38,2 %", BEST)],
], tableOpts(M, 1.45, CW, [3.0, 4.9, 2.6, CW - 10.5], 17));
const tarjetas = [
  ["Factores separados", "Científicos: celda y objetivo. De ruido: lr, dropout y optimizador, ajustados para cada celda (tuning playbook de Google)."],
  ["Tres semillas y regla fija", "Semillas 42, 7 y 2026 con enable_op_determinism. Gana el menor MAE medio de validación, regla fijada antes de entrenar."],
  ["Todo registrado", "MLflow guarda parámetros, métricas, pérdidas por época y artefactos de las 150 corridas."],
];
tarjetas.forEach((t, i) => {
  const x = M + i * ((CW - 0.4) / 3 + 0.2), w = (CW - 0.4) / 3;
  card(s, x, 4.2, w, 2.5);
  numCircle(s, x + 0.22, 4.4, i + 1, C.accent2);
  s.addText(t[0], { x: x + 0.78, y: 4.4, w: w - 1.0, h: 0.42, fontSize: 16, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
  s.addText(t[1], { x: x + 0.22, y: 5.0, w: w - 0.44, h: 1.55, fontSize: 16, color: C.text1, valign: "top", margin: 0, isTextBox: true });
});

// ---------- 6. Hiperparámetros ----------
pres.addSection({ title: "Resultados" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
s.addText("Hiperparámetros: manda la regularización", { placeholder: "title" });
s.addImage({ path: FIG("fig3_malla_hiperparametros.png"), x: M, y: 1.3, w: CW, h: CW / 4.062, altText: "Malla de hiperparámetros, MAE de validación" });
caption(s, "Ronda 3: MAE de validación (miles), media de 3 semillas, por tasa de aprendizaje y dropout.", M, 1.3 + CW / 4.062 + 0.02, CW);
const kw = (CW - 3 * 0.2) / 4;
stat(s, M, 4.65, kw, 1.45, "lr", "el factor con más efecto: 5.861 con 0,0003 y 6.249 con 0,003");
stat(s, M + (kw + 0.2), 4.65, kw, 1.45, "dropout", "el error baja al subirlo: 6.117 con 0 y 5.937 con 0,4");
stat(s, M + 2 * (kw + 0.2), 4.65, kw, 1.45, "optimizador", "casi no importa: Adam 6.018 y RMSprop 6.043");
stat(s, M + 3 * (kw + 0.2), 4.65, kw, 1.45, "GRU", "5.653 ± 148, frente a 5.780 ± 280 de la LSTM", C.accent2, "E3F1EC");
s.addText("Las cinco mejores configuraciones están dentro de una desviación estándar (5.653 a 5.760). Lo que se mantiene es GRU, dropout 0,4 y objetivo de cambio, más que un valor exacto de lr.",
  { x: M, y: 6.25, w: CW, h: 0.6, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });

// ---------- 7. Curvas ----------
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
s.addText("Curvas de aprendizaje", { placeholder: "title" });
const fw = 8.5;
s.addImage({ path: FIG("fig2_curvas_aprendizaje.png"), x: M, y: 1.35, w: fw, h: fw / 2.188, altText: "Curvas de pérdida y MAE por época" });
caption(s, "Continua: entrenamiento. Discontinua: validación. Punteadas: persistencia y deriva. Una línea por semilla.", M, 1.35 + fw / 2.188 + 0.02, fw);
const cx = M + fw + 0.3, cw2 = CW - fw - 0.3;
const curvas = [
  ["Objetivo de nivel (E4)", "Mejor época entre 22 y 53 según la semilla. El dropout altera un estado de unos 800 mil."],
  ["Objetivo de cambio", "Mínimo en la época 2; después la validación sube y EarlyStopping recupera esos pesos."],
  ["Validación por debajo", "Por el dropout en entrenamiento y porque el sábado cambia menos (varianza al 48 %)."],
];
curvas.forEach((t, i) => {
  const y = 1.35 + i * 1.62;
  card(s, cx, y, cw2, 1.47);
  s.addText([{ text: t[0], options: { bold: true, color: C.text2, breakLine: true, fontSize: 15 } }, { text: t[1], options: { fontSize: 14.5 } }],
    { x: cx + 0.18, y: y + 0.12, w: cw2 - 0.36, h: 1.25, color: C.text1, valign: "top", margin: 0, isTextBox: true });
});
card(s, M, 5.65, fw, 0.95, "E3F1EC");
s.addText([{ text: "Sin dropout, R² del cambio: ", options: { bold: true, color: C.accent2 } },
  { text: "0,64 en entrenamiento y 0,60 en validación con la GRU final (E4: 0,30 y 0,45). En la época elegida no hay sobreajuste." }],
  { x: M + 0.2, y: 5.7, w: fw - 0.4, h: 0.85, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true });

// ---------- 8. Prueba ----------
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
s.addText("Prueba: −36,8 % frente a la persistencia", { placeholder: "title" });
stat(s, M, 1.35, kw, 1.4, "5.835", "MAE de la GRU en prueba (domingo)", C.accent2, "E3F1EC");
stat(s, M + (kw + 0.2), 1.35, kw, 1.4, "−36,8 %", "frente a la persistencia; IC [−5.503; −902], p < 0,001", C.accent2, "E3F1EC");
stat(s, M + 2 * (kw + 0.2), 1.35, kw, 1.4, "−12,7 %", "frente a Ridge con las mismas entradas; p < 0,001", C.accent2, "E3F1EC");
stat(s, M + 3 * (kw + 0.2), 1.35, kw, 1.4, "−12,8 %", "frente a la deriva; IC [−1.687; +205], p = 0,06", C.accent3, "FBF3E4");
const R = (a, b, c, d, best) => best ? [{ text: a, options: BEST }, N(b, BEST), N(c, BEST), N(d, BEST)] : [a, N(b), N(c), N(d)];
s.addTable([
  [HDR("Modelo"), { text: "MAE", options: { bold: true, color: C.text2, fill: { color: C.background2 }, align: "right" } },
    { text: "RMSE", options: { bold: true, color: C.text2, fill: { color: C.background2 }, align: "right" } },
    { text: "MAPE", options: { bold: true, color: C.text2, fill: { color: C.background2 }, align: "right" } }],
  R("Persistencia", "9.230", "11.584", "1,6 %"),
  R("Avance: E5 LSTM (nivel)", "9.906", "12.882", "—"),
  R("Ronda 1: E4 GRU (nivel)", "7.370", "10.194", "1,1 %"),
  R("Deriva", "6.687", "8.944", "1,0 %"),
  R("Ridge (cambio)", "6.687", "8.306", "1,1 %"),
  R("Mejor LSTM (ronda 3)", "6.022", "7.667", "1,0 %"),
  R("GRU seleccionada", "5.835", "7.364", "1,0 %", true),
], tableOpts(M, 2.95, 7.1, [3.5, 1.2, 1.2, 1.2], 14));
bullets(s, [
  "Una sola evaluación, con las mismas 152 ventanas para todos los modelos.",
  "IC por bootstrap de bloques de 12 pasos; Diebold-Mariano con varianza de Newey-West.",
  "Validación 5.599 y prueba 5.835: 4 % de diferencia.",
  "Frente a la deriva la diferencia no es significativa con un solo día de prueba, así que no afirmamos que la red sea mejor.",
], { x: 8.0, y: 3.0, w: 4.73, h: 3.7, size: 15, gap: 10 });

// ---------- 9. Errores ----------
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
s.addText("Errores: dónde gana y dónde falla", { placeholder: "title" });
const eh = 5.25;
s.addImage({ path: FIG("fig5_errores_prueba.png"), x: M, y: 1.35, w: eh * 1.463, h: eh, altText: "Pronóstico, residuos y MAE por franja horaria en prueba" });
const ex = M + eh * 1.463 + 0.35, ew = W - M - ex;
s.addTable([
  [HDR("Tercil del cambio"), { text: "GRU", options: { bold: true, color: C.text2, fill: { color: C.background2 }, align: "right" } },
    { text: "Persist.", options: { bold: true, color: C.text2, fill: { color: C.background2 }, align: "right" } }],
  ["Estable", N("5.213"), N("2.514")],
  ["Medio", N("5.180"), N("7.935")],
  [{ text: "Rampa", options: BEST }, N("7.098", BEST), N("17.215", BEST)],
], tableOpts(ex, 1.4, ew, [ew - 2.3, 1.1, 1.2], 14));
bullets(s, [
  "En las rampas el error baja 59 %; en la madrugada, de 9.815 a 3.883.",
  "Si la serie está estable, la persistencia acierta y la red predice un cambio que no ocurre.",
  "Sesgo de −2.633: aprendió con días laborales y el domingo tiene 17 % menos actividad.",
  "Ljung-Box p = 0,79; 0,28 y 0,21: no queda autocorrelación en los residuos.",
], { x: ex, y: 3.25, w: ew, h: 3.5, size: 16, gap: 10 });

// ---------- 10. Riesgos y uso ----------
pres.addSection({ title: "Cierre" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Cierre" });
s.addText("Riesgos y recomendaciones de uso", { placeholder: "title" });
const colw = (CW - 0.3) / 2;
card(s, M, 1.35, colw, 5.35, "FBF3E4");
s.addText("Limitaciones y riesgos", { x: M + 0.3, y: 1.55, w: colw - 0.6, h: 0.45, fontFace: THEME.headFontFace, fontSize: 21, bold: true,
  color: C.accent3, margin: 0, isTextBox: true });
bullets(s, [
  "Una sola semana: no hay evidencia entre semanas ni en festivos.",
  "Un solo día de prueba: el intervalo no captura la variación entre días.",
  "El índice depende de la cobertura (r = 0,88) y no mide bytes.",
  "El avance ya había usado la prueba; esta versión no la usó para decidir.",
  "Sobrestimar el fin de semana puede dejar recursos encendidos sin necesidad.",
], { x: M + 0.3, y: 2.15, w: colw - 0.6, h: 4.4, size: 17.5, gap: 14, labelColor: C.accent3 });
const x2 = M + colw + 0.3;
card(s, x2, 1.35, colw, 5.35, "E3F1EC");
s.addText("Uso y despliegue", { x: x2 + 0.3, y: 1.55, w: colw - 0.6, h: 0.45, fontFace: THEME.headFontFace, fontSize: 21, bold: true,
  color: C.accent2, margin: 0, isTextBox: true });
bullets(s, [
  ["Uso.", "Actividad agregada de la ciudad a 10 minutos; no por celda ni por usuario."],
  ["Respaldo.", "La deriva: no cuesta nada y su MAE es solo 15 % mayor."],
  ["Monitoreo.", "MAE móvil diario frente a la deriva y alerta por sesgo."],
  ["Antes de producción.", "Más semanas, día de la semana y cobertura como entradas, TimeSeriesSplit e intervalos de predicción."],
], { x: x2 + 0.3, y: 2.15, w: colw - 0.6, h: 4.4, size: 17.5, gap: 14, labelColor: C.accent2 });

// ---------- 11. Conclusiones ----------
s = pres.addSlide({ masterName: "CIERRE", sectionTitle: "Cierre" });
s.addText("Conclusiones y reproducibilidad", { placeholder: "title" });
const concl = [
  "Cambiar el objetivo a predecir el cambio importó más que la arquitectura: pasamos de 9.906 a 5.835 de MAE.",
  "La GRU supera con significancia a la persistencia (−36,8 %) y a Ridge (−12,7 %). Frente a la deriva, la ventaja de 12,8 % no es significativa.",
  "Su valor está en las subidas y bajadas de actividad. Su debilidad es el sesgo del fin de semana, que requiere más semanas de datos.",
];
concl.forEach((t, i) => {
  const y = 1.5 + i * 1.6;
  numCircle(s, M, y + 0.05, i + 1, "CADCFC", C.text2);
  s.addText(t, { x: M + 0.65, y, w: 6.6, h: 1.4, fontSize: 17, color: C.background1, valign: "top", margin: 0, isTextBox: true });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.3, y: 1.5, w: W - M - 8.3, h: 5.0, rectRadius: 0.08, fill: { color: C.accent6 }, line: { color: C.accent6, width: 0 } });
s.addText("Repositorio", { x: 8.55, y: 1.7, w: W - M - 8.8, h: 0.45, fontFace: THEME.headFontFace, fontSize: 20, bold: true, color: C.background1, margin: 0, isTextBox: true });
bullets(s, [
  "Notebook ejecutado de principio a fin (unos 6 minutos en CPU)",
  "Pesos .keras de las 3 semillas y metadatos",
  "MLflow: 151 corridas registradas",
  "requirements_u4.txt y semillas 42, 7 y 2026",
  "SHA-256 del CSV verificado al cargar",
  "Los pesos recargados reproducen el resultado (diferencia de 0,0000)",
], { x: 8.55, y: 2.3, w: W - M - 8.8, h: 3.7, size: 15, gap: 8, color: C.background1 });
s.addText("github.com/dsiriarte/Tecnicas-Avanzadas-de-Modelado-en-IA/tree/main/Unidad 4", { x: M, y: 6.75, w: 7.4, h: 0.35, fontSize: 12,
  color: "CADCFC", margin: 0, isTextBox: true, hyperlink: { url: "https://github.com/dsiriarte/Tecnicas-Avanzadas-de-Modelado-en-IA/tree/main/Unidad%204" } });

(async () => {
  await pres.writeFile({ fileName: SALIDA });
  const tema = process.argv[2];
  if (tema) {
    const { applyTheme } = require(tema);
    await applyTheme(SALIDA, THEME);
  }
  console.log("Escrito", SALIDA);
})();
