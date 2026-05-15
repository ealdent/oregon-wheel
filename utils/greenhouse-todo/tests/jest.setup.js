document.body.innerHTML = `
    <div id="crosshair"></div>
    <div id="hover-tooltip"></div>
    <div id="blocker">
        <div id="instructions"></div>
    </div>
    <div id="ui-container"></div>
    <div id="add-todo-modal">
        <form id="add-todo-form">
            <input type="text" id="todo-title">
            <textarea id="todo-desc"></textarea>
            <select id="todo-urgency"></select>
            <button type="submit"></button>
        </form>
        <span id="close-add-modal"></span>
    </div>
    <div id="todo-modal">
        <span id="close-modal"></span>
        <h2 id="modal-title"></h2>
        <p id="modal-desc"></p>
        <span id="modal-health"></span>
        <span id="modal-urgency"></span>
        <span id="modal-status"></span>
        <button id="btn-status-procrastinating"></button>
        <button id="btn-status-inprogress"></button>
        <button id="btn-status-almostdone"></button>
        <select id="todo-effort"></select>
        <button id="btn-checkin"></button>
        <button id="btn-complete"></button>
    </div>
    <div id="mobile-controls">
        <div id="look-zone"></div>
        <div id="joystick"><div id="stick"></div></div>
        <button id="mobile-menu-btn"></button>
    </div>
`;
