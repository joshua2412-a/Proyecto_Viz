/* ==========================================================================
   Gráficos fuera de la vista: se dibujan cuando hacen falta

   Dash carga solo todo lo que esté en /assets: este archivo no se importa
   desde ninguna parte.

   Por qué: al entrar en una pestaña, React monta todos sus dcc.Graph a la vez
   y Plotly los dibuja todos en el mismo golpe. Medido en el navegador del
   autor, Resultados (9 gráficos) congelaba la página 1,3 s y Predicción (7)
   casi 1 s: el clic parecía no responder y las partículas se paraban a mitad
   del viaje.

   Qué hace: se interpone en Plotly.react, que es lo que llama dcc.Graph para
   dibujar. Si el gráfico aún no se ha dibujado nunca y queda lejos de la
   pantalla, no se dibuja todavía: se reserva su alto (para que la página no
   salte después) y se dibuja
     - cuando se acerca a la vista (a 1.000 px, antes de que llegue), o
     - en los ratos libres del navegador, uno cada vez, a partir de 1,6 s de
       entrar en la pestaña (cuando ya terminó el viaje de las partículas).
   Lo que ya está dibujado (por ejemplo, Predicción al pulsar Predecir) se
   actualiza al momento, como siempre.

   dcc.Graph espera la promesa que devuelve Plotly.react para enlazar sus
   eventos: se le devuelve una que se cumple cuando el gráfico se dibuja de
   verdad.
   ========================================================================== */

(function () {
  "use strict";

  var MARGEN = 1000;          // px: se dibuja antes de entrar en la vista
  var ESPERA_LIBRE = 1600;    // ms tras el último aplazamiento
  var pendientes = [];
  var ultimo = 0, temporizador = null, observador = null;

  function cerca(gd) {
    var r = gd.getBoundingClientRect();
    if (!r.width && !r.height && !gd.offsetParent) return false; // oculto (subpestaña)
    return r.top < window.innerHeight + MARGEN * 0.25 && r.bottom > -MARGEN;
  }

  function dibujar(P, original, gd) {
    var p = gd.__pendiente;
    if (!p) return;
    gd.__pendiente = null;
    delete gd.dataset.diferido;
    gd.style.minHeight = p.minimo;
    if (observador) observador.unobserve(gd);
    var i = pendientes.indexOf(gd);
    if (i >= 0) pendientes.splice(i, 1);
    if (!gd.isConnected) { p.cumplir(); return; }   // la pestaña ya se fue
    try {
      Promise.resolve(original.apply(P, p.args)).then(p.cumplir, p.fallar);
    } catch (e) { p.fallar(e); }
  }

  function vaciarEnRatosLibres(P, original) {
    clearTimeout(temporizador);
    temporizador = setTimeout(function paso() {
      if (!pendientes.length) return;
      if (performance.now() - ultimo < ESPERA_LIBRE) { temporizador = setTimeout(paso, 300); return; }
      var pedir = window.requestIdleCallback || function (f) { return setTimeout(f, 50); };
      pedir(function () {
        // Primero los visibles en su pestaña; los de subpestañas ocultas
        // esperan a que se muestren (los dibuja el observador)
        var gd = null;
        for (var k = 0; k < pendientes.length; k++) {
          if (pendientes[k].offsetParent && pendientes[k].isConnected) { gd = pendientes[k]; break; }
        }
        pendientes = pendientes.filter(function (g) { return g.isConnected; });
        if (gd) dibujar(P, original, gd);
        if (pendientes.some(function (g) { return g.offsetParent; })) temporizador = setTimeout(paso, 120);
      }, { timeout: 2000 });
    }, ESPERA_LIBRE);
  }

  function parchear(P) {
    if (!P || P.__diferido || typeof P.react !== "function") return;
    var original = P.react;
    P.__diferido = true;
    if ("IntersectionObserver" in window) {
      observador = new IntersectionObserver(function (entradas) {
        entradas.forEach(function (e) { if (e.isIntersecting) dibujar(P, original, e.target); });
      }, { rootMargin: MARGEN + "px 0px" });
    }
    P.react = function (gd) {
      var args = arguments;
      var aplazable = gd && gd.nodeType === 1 && !gd._fullLayout && gd.closest &&
                      gd.closest("#contenido-pestana") && !cerca(gd);
      if (!aplazable && !(gd && gd.__pendiente)) return original.apply(P, args);
      // Ya aplazado: se queda con la figura más reciente
      if (gd.__pendiente) { gd.__pendiente.args = args; return gd.__pendiente.promesa; }
      var p = { args: args, minimo: gd.style.minHeight };
      p.promesa = new Promise(function (cumplir, fallar) { p.cumplir = cumplir; p.fallar = fallar; });
      gd.__pendiente = p;
      gd.dataset.diferido = "1";
      // Reservar el alto que tendrá, para que la página no salte al dibujarlo
      var fig = args[1] || {}, alto = fig.layout && fig.layout.height;
      if (!gd.offsetHeight) gd.style.minHeight = (alto || 450) + "px";
      pendientes.push(gd);
      ultimo = performance.now();
      if (observador) observador.observe(gd);
      vaciarEnRatosLibres(P, original);
      return p.promesa;
    };
  }

  // Plotly llega más tarde (lo carga dcc.Graph): se parchea en cuanto existe
  if (window.Plotly) parchear(window.Plotly);
  else {
    var intentos = 0, espera = setInterval(function () {
      if (window.Plotly) { parchear(window.Plotly); clearInterval(espera); }
      else if (++intentos > 600) clearInterval(espera);
    }, 50);
  }
})();
