
$(document).ready(function() {
    $('a').live('click',sendMess);
    setInterval(camRefresh,60000);
});

function sendMess(e) {
    //if (!window.parent) return;
    if (!$(this).hasClass('arrow')) return;
    e.preventDefault();
    url = $(this).attr('href');
    window.parent.postMessage('url|'+url,'*');
}

function camRefresh() {
    $.ajax({
        'url':"?ajax",
        'success': function(data) {
            $('#cams').html(data);
        }
    });
}
