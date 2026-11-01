/*
 * Asistente virtual: abrir, cerrar y acompañar la conversación.
 *
 * Todo lo que es solo interfaz va aquí y no en callbacks de Dash: abrir el
 * panel es cambiar una clase, y pedírselo al servidor añadía un viaje de ida y
 * vuelta antes de que el panel se moviera. Las respuestas sí las calcula el
 * servidor (utils/asistente_panel.py).
 *
 * El panel y el botón se montan cuando React pinta el layout, después de que
 * cargue este archivo, así que se escucha en el documento (delegación) en vez
 * de buscar los elementos al arrancar.
 */
(function () {
  "use strict";

  function raiz() { return document.getElementById("asistente"); }
  function abierto() { var r = raiz(); return !!(r && r.classList.contains("asistente-abierto")); }

  // Lleva la conversación a la última respuesta. Si es más alta que el
  // panel, se muestra desde su principio (con la pregunta encima): bajar
  // hasta el final obligaría a subir para empezar a leerla.
  function alFinal(suave) {
    var lista = document.getElementById("asistente-mensajes");
    if (!lista) return;
    var destino = lista.scrollHeight;
    var ultimo = lista.lastElementChild;
    if (ultimo && ultimo.offsetHeight > lista.clientHeight - 60) {
      var anterior = ultimo.previousElementSibling || ultimo;
      destino = lista.scrollTop + anterior.getBoundingClientRect().top
        - lista.getBoundingClientRect().top - 12;
    }
    lista.scrollTo({ top: destino, behavior: suave ? "smooth" : "auto" });
  }

  function abrir() {
    var r = raiz();
    if (!r) return;
    ocultarPista(false);
    r.classList.add("asistente-abierto");
    var boton = document.getElementById("asistente-abrir");
    var panel = document.getElementById("asistente-panel");
    if (boton) boton.setAttribute("aria-expanded", "true");
    if (panel) panel.setAttribute("aria-hidden", "false");
    alFinal(false);
    // En pantallas táctiles no se enfoca el campo: abriría el teclado y
    // taparía media conversación antes de que el usuario decida escribir.
    var tactil = window.matchMedia && window.matchMedia("(hover: none)").matches;
    var entrada = document.getElementById("asistente-entrada");
    // dcc.Input no admite aria-label en Python: se pone aquí
    if (entrada) entrada.setAttribute("aria-label", "Tu pregunta para el asistente");
    if (entrada && !tactil) setTimeout(function () { entrada.focus({ preventScroll: true }); }, 180);
  }

  function cerrar(devolverFoco) {
    var r = raiz();
    if (!r) return;
    r.classList.remove("asistente-abierto");
    var boton = document.getElementById("asistente-abrir");
    var panel = document.getElementById("asistente-panel");
    if (boton) boton.setAttribute("aria-expanded", "false");
    if (panel) panel.setAttribute("aria-hidden", "true");
    if (devolverFoco && boton) boton.focus({ preventScroll: true });
  }

  document.addEventListener("click", function (e) {
    var objetivo = e.target;
    if (!(objetivo instanceof Element)) return;
    // El botón flotante solo abre (con el panel abierto se oculta y el panel
    // tiene su propio cierre). Si alternara, un doble clic lo cerraría al
    // instante y parecería que no hace nada.
    if (objetivo.closest("#asistente-abrir")) {
      if (!abierto()) abrir();
      return;
    }
    // «Pregúntale a Glía» junto a un gráfico: el servidor responde por el
    // gráfico (asistente_panel.conversar); aquí solo se abre el panel.
    if (objetivo.closest(".glia-chip")) {
      ocultarPista(true);
      if (!abierto()) abrir();
      return;
    }
    // La pista: su texto abre el panel y hace la pregunta (la primera
    // sugerencia de la pestaña); la ✕ la descarta.
    if (objetivo.closest(".glia-pista-texto")) {
      ocultarPista(true);
      abrir();
      setTimeout(function () {
        var primera = document.querySelector(".asistente-sugerencia");
        if (primera) primera.click();
      }, 120);
      return;
    }
    if (objetivo.closest(".glia-pista-cerrar")) {
      ocultarPista(true);
      return;
    }
    if (objetivo.closest("#asistente-cerrar")) {
      cerrar(true);
      return;
    }
    // En el móvil el panel ocupa casi toda la pantalla: al ir a una pestaña
    // se cierra para que se vea a dónde se ha llegado.
    if (objetivo.closest(".asistente-ir") && window.innerWidth < 576) cerrar(false);
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && abierto()) {
      // Si el índice está abierto, Escape es suyo (indice.js).
      var indice = document.getElementById("indice");
      if (indice && indice.open) return;
      cerrar(true);
    }
  });

  // Cada mensaje nuevo lleva la conversación a la última respuesta. Se
  // observa la raíz del asistente (estable) y no la lista, que Dash puede
  // volver a montar; y no el documento entero, que durante la salida de la
  // portada cambia sin parar. Solo actúa si el cambio es en la lista.
  (function vigilarMensajes() {
    var raizA = document.getElementById("asistente");
    if (!raizA) { setTimeout(vigilarMensajes, 300); return; }
    new MutationObserver(function (cambios) {
      for (var i = 0; i < cambios.length; i++) {
        var t = cambios[i].target;
        if (t.id === "asistente-mensajes") {
          // Tras pintar: las fórmulas de MathJax cambian la altura después
          requestAnimationFrame(function () { alFinal(true); });
          setTimeout(function () { alFinal(true); }, 350);
          return;
        }
      }
    }).observe(raizA, { childList: true, subtree: true });
  })();

  // ---- La pista de cada pestaña (A4) ---------------------------------------
  // Glía ofrece una pregunta útil la primera vez que se entra a una pestaña:
  // sale junto al botón 1,5 s después de llegar, se va sola a los 9 s y no
  // vuelve en la misma visita (sessionStorage). Nunca con el panel abierto.
  var temporizadores = [];
  function pistaEl() { return document.querySelector(".glia-pista"); }
  function yaVista(texto) {
    try { return sessionStorage.getItem("glia-pista:" + texto) === "1"; } catch (e) { return false; }
  }
  function marcarVista(texto) {
    try { sessionStorage.setItem("glia-pista:" + texto, "1"); } catch (e) { /* sin almacenamiento */ }
  }
  function ocultarPista(descartar) {
    temporizadores.forEach(clearTimeout);
    temporizadores = [];
    var p = pistaEl();
    if (p) p.classList.remove("glia-pista-visible");
    if (descartar) {
      var t = document.getElementById("glia-pista");
      if (t && t.textContent) marcarVista(t.textContent);
    }
  }
  var ultimaPista = "";
  function programarPista() {
    var t = document.getElementById("glia-pista");
    var texto = t ? t.textContent : "";
    if (texto === ultimaPista) return;
    ultimaPista = texto;
    ocultarPista(false);
    if (!texto || yaVista(texto)) return;
    temporizadores.push(setTimeout(function () {
      var r = raiz();
      if (abierto() || !r || r.classList.contains("glia-oculta")) return;
      var p = pistaEl();
      if (!p) return;
      p.classList.add("glia-pista-visible");
      marcarVista(texto);
      temporizadores.push(setTimeout(function () { ocultarPista(false); }, 9000));
    }, 1500));
  }
  // Se observa solo el botón de la pista: observar el texto de todo el
  // documento despertaba este código en cada fotograma de los contadores.
  (function vigilarPista() {
    var t = document.getElementById("glia-pista");
    if (!t) { setTimeout(vigilarPista, 300); return; }
    new MutationObserver(programarPista).observe(t, { childList: true, subtree: true, characterData: true });
    programarPista();
  })();

  window.asistenteAbrir = abrir;
})();
