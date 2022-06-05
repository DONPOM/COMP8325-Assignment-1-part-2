
    __compete_code = 'a1fad508ef46d5362bbcd5f91b845c03';
    (function () {
        var s = document.createElement('script'),
            d = document.getElementsByTagName('head')[0] ||
                document.getElementsByTagName('body')[0],
            t = 'https:' == document.location.protocol ? 
                'https://c.compete.com/bootstrap/' : 
                'http://c.compete.com/bootstrap/';
        s.src = t + __compete_code + '/bootstrap.js';
        s.type = 'text/javascript';
        s.async = 'async'; 
        if (d) { d.appendChild(s); }
    }());
    