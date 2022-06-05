
function xCnt(write) {
    var rf = ""+escape(document.referrer);
    var res = "<img height='1' width='1' border='0' alt='' src='http://cnt2.xhamster.com/cnt.php?rf="+rf+"&srv=m24'>";
    if (write) document.write(res);
    else $('body').append(res);
}
xCnt(true);