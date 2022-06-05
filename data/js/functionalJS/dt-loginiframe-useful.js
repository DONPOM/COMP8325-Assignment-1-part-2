
//<![CDATA[
if (typeof(DB) == "undefined")
    DB = {};
    
DB.ShowLogin = function(obj){
    var overlay = Overlay.Get({
        onclick: function(obj){
            $('loginbox').hide();
        }
    });
    overlay.setStyle({ opacity: .5 }); 
    $('loginbox').down('div.loginboxFrame').down('div.loginboxiframe').update('<iframe src="https://secure.donbest.com/site3/sbrlogin.html" width="100%" frameborder="0" scrolling="no" style="background: #FFF;">No Frames to login</iframe>');
    overlay.show();
    if (document.body)
        document.body.appendChild($('loginbox'));
    $('loginbox').show();
}
//]]>
