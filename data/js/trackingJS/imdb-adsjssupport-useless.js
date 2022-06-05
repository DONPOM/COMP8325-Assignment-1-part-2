
            // ensures js doesn't die if ads service fails.  
            // Note that we need to define the js here, since ad js is being rendered inline after this.
            (function(f) {
                // Fallback javascript, when the ad Service call fails.  
                
                if((window.csm === undefined || window.generic === undefined || window.consoleLog === undefined)) {
                    if (console !== undefined && console.log !== undefined) {
                        console.log("one or more of window.csm, window.generic or window.consoleLog has been stubbed...");
                    }
                }
                
                window.csm = window.csm || { measure:f, record:f, duration:f, listen:f, metrics:{} };
                window.generic = window.generic || { monitoring: { start_timing: f, stop_timing: f } };
                window.consoleLog = window.consoleLog || f;
            })(function() {});
        