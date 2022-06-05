

            function MSOLayout_GetRealOffset(StartingObject, OffsetType, EndParent) {

                var realValue = 0;

                if (!EndParent) EndParent = document.body;

                for (var currentObject = StartingObject; currentObject && currentObject != EndParent && currentObject != document.body; currentObject = currentObject.offsetParent) {

                    var offset = eval('currentObject.offset' + OffsetType);

                    if (offset) realValue += offset;

                }

                return realValue;

            }


        