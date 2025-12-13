#import <Foundation/Foundation.h>

@class NSArray;

@interface CGVirtualDisplaySettings : NSObject {
  NSArray *_modes;
  unsigned int _hiDPI;
}

@property(nonatomic) unsigned int hiDPI;
- (void)dealloc;
- (id)init;
@property(retain, nonatomic) NSArray *modes;

@end
