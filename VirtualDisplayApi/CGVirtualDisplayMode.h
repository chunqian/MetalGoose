#import <Foundation/Foundation.h>

@interface CGVirtualDisplayMode : NSObject {
  unsigned int _width;
  unsigned int _height;
  double _refreshRate;
}

@property(readonly, nonatomic) double refreshRate;
@property(readonly, nonatomic) unsigned int height;
@property(readonly, nonatomic) unsigned int width;
- (id)initWithWidth:(unsigned int)arg1
             height:(unsigned int)arg2
        refreshRate:(double)arg3;

@end
