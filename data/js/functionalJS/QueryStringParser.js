/*
 * DPN 07/29/2014
 * Simple tool to parse the querystring into an array of KeyValuePair objects.
 *
 *
 * Usage:
 *
 * // parse the querystring into an array of KeyValuePair objects...
 * var arKeyValuePairs = new QueryStringParser().parse();
 *
 * // parse the querystring, retrieve the value associated with the key "keyword", and place that value into a textbox...
 * $("#txtKeyword").val((new QueryStringParser().getValue("keyword")));
 */


function QueryStringParser() {
    var self = this;

    self.keyValuePairs = null;


    // Parses the querystring into an array of KeyValuePair objects.
    self.parse = function () {

        var s1 = document.location.href;
        var s2 = window.location.href;
        var url = (s1.length > s2.length ? s1 : s2);                                // interesting bug: we've found that some mobile browsers truncate one or the other...check the length, take the longer of the two

        var ar1 = null, kvps = null;

        if (self.keyValuePairs == null) {
            self.keyValuePairs = [];

            if (url.indexOf("?") > 0) {
                ar1 = url.split("?");                                               // discard all but the querystring itself

                if (ar1.length > 1) {
                    kvps = ar1[1].split("&");                                       // split querystring into key/value pairs

                    for (var i = 0; i < kvps.length; i++) {
                        self.keyValuePairs.push(new KeyValuePair(kvps[i]));         // create keyValuePair object for this pair
                    }
                }
            }
            else {
                self.keyValuePairs = null;
            }
        }

        return self.keyValuePairs;

    };


    // Returns a value from the parsed querystring corresponding to the given key.
    self.getValue = function(key) {

        var value = "";
        var keyUpper = key.toUpperCase();

        if (self.keyValuePairs == null) {
            self.parse();
        }


        if (self.keyValuePairs != null) {

            for (var i = 0; i < self.keyValuePairs.length; i++) {
                if (self.keyValuePairs[i].key.toUpperCase() == keyUpper) {
                    value = self.keyValuePairs[i].value;
                    break;
                }
            }
        }

        return decodeURIComponent(value.replace(/\+/g, " "));

    };


    // Returns a value from the parsed querystring by index.
    self.getValueByIndex = function(index) {

        var value = null;

        if (self.keyValuePairs == null) {
            self.parse();
        }

        if (self.keyValuePairs != null) {
            value = self.keyValuePairs[i].value;
        }

        return value;

    };

}


function KeyValuePair(data) {
    var self = this;

    self.key = "";
    self.value = "";

    if (data.length > 0 && data.indexOf("=") > 0) {
        var ar = data.split("=");
        self.key = ar[0];
        self.value = ar[1];
    }
}

