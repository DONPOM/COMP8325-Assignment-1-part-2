(function() {
    function trk(a) {
        if (a) {
            var img = new Image();
            img.onload = function() {
                img = null
            };
            img.src = a;
        }
    };

    function displayTL1(code) {
        var location = document.getElementById("TL1_location") || null;
        if (location) {
            location.innerHTML = code;
        } else {
            document.write(code);
        }
    }
    var code = '<style>' +
    '	.ad-tl2b {text-align:left;font-size:70%; margin: 0px; font-family: Helvetica,Arial,sans-serif;}' +
    '	.ad-tl2b p { margin:0px 0px 4px 0px !important; line-height: 120%; color: #1D1DA3;}' +
    '	.ad-tl2b p .y-fp-pg-controls {display:block;background-position:left -2275px; }' +
    '	.ad-tl2b a { padding-top:0px; color: #1D1DA3; margin: 0px;}' +
    '	.ad-tl2b b { font-weight:700; font-size: 9px; text-align: left; margin: 0px;}' +
    '	.ad-tl2b #tl1_slug {text-align:left; font-size:90%; color: #B3B3B5; font-weight:400;padding: 0 0 3px 1px; margin: 0.55em 0;}' +
    '	.ad-tl2b a:link, .ad-tl2b a:visited { text-decoration: none;}' +
    '	.ad-tl2b a:hover { text-decoration: underline;}' +
    '	.ad-tl2b img { border: 0 none; vertical-align: bottom;}' +
    '</style>' +
    '<div class="ad-tl2b small">' +
    '<p id="tl1_slug">Sponsored</p>' + '	<p class="y-fp-pg-controls"><a href="https://ads.yahoo.com/clk?3,eJytjdGKgzAQRb-mb1aSJgZD2IeYqCzVuEJa0DdNqtZ2a6GWZfv1VVy2P9DDZZi5c-FCxEANmgbaxtQEYVoDBjE9YL-itjYOYIwhjCghlEDinEwihYW9yMV2HRcBn0k8Y3.4Qssjo.zF..LnKeU-OMp2eV-KteZvQeyHPP.bP-de3kqfl-Vm64Wn4D8mbZf2O1w8WpxoA5QMx1RHZyWAl-nQmzxcanVUmqMsTnH2KvhwnG4cr7cV4qtNNKm6u79VNwyuGb6n8wmgh1dW,http://au.tv.yahoo.com/plus7/?cmp=mt" role="presentation" target="_blank"><img src="https://s.yimg.com/cv/ae/au/audience/111107/120x45ukfmb7f08.jpg" width="120" height="45" border="0" alt="" /></a></p>' +
    '	<p><a href="https://ads.yahoo.com/clk?3,eJytjdGKgzAQRb-mb1aSJgZD2IeYqCzVuEJa0DdNqtZ2a6GWZfv1VVy2P9DDZZi5c-FCxEANmgbaxtQEYVoDBjE9YL-itjYOYIwhjCghlEDinEwihYW9yMV2HRcBn0k8Y3.4Qssjo.zF..LnKeU-OMp2eV-KteZvQeyHPP.bP-de3kqfl-Vm64Wn4D8mbZf2O1w8WpxoA5QMx1RHZyWAl-nQmzxcanVUmqMsTnH2KvhwnG4cr7cV4qtNNKm6u79VNwyuGb6n8wmgh1dW,http://au.tv.yahoo.com/plus7/?cmp=mt" target="_blank">Catch-up and watch<br />full episodes of your<br />favourite TV shows.<br /></a></p>' +
    '</div>';

    displayTL1(code);


}());

