/* ==========================================================================
   Índice desplegable

   Dash carga solo todo lo que este en /assets, asi que este archivo no se
   importa desde ninguna parte: basta con que exista.

   El índice es un <details> nativo, que ya se abre y se cierra solo. Esto
   añade lo que un desplegable necesita y <details> no trae: cerrarse al
   elegir una pestaña, al pulsar Escape y al hacer clic fuera de él. El cambio
   de pestaña en sí lo hace el callback `navegar_con_indice` de app.py.
   ========================================================================== */

(function () {
  "use strict";

  function indice() {
    return document.getElementById("indice");
  }

  document.addEventListener("click", function (evento) {
    var d = indice();
    if (!d || !d.open) return;
    if (evento.target.closest(".indice-item") || !d.contains(evento.target)) {
      d.open = false;
    }
  });

  document.addEventListener("keydown", function (evento) {
    var d = indice();
    if (evento.key !== "Escape" || !d || !d.open) return;
    d.open = false;
    d.querySelector("summary").focus();
  });
})();
