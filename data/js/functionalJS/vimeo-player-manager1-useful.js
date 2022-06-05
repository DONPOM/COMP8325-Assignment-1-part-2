
        function onVimeoDomReady(e) {
            
                            
                PlayerManager.dispatchReadySignals();
            
                    }

                    if ('addEvent' in window) {
                window.addEvent('domready', onVimeoDomReady);
            }
            else {
                onVimeoDomReady();
            }
            