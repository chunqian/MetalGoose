#import "CDStructures.h"
#import <Foundation/Foundation.h>

@interface CGVirtualDisplayDescriptor : NSObject {
  unsigned int _vendorID;
  unsigned int _productID;
  unsigned int _serialNum;
  NSString *_name;
  struct CGSize _sizeInMillimeters;
  unsigned int _maxPixelsWide;
  unsigned int _maxPixelsHigh;
  struct CGPoint _redPrimary;
  struct CGPoint _greenPrimary;
  struct CGPoint _bluePrimary;
  struct CGPoint _whitePoint;
  dispatch_queue_t _queue;
  CDUnknownBlockType _terminationHandler;
}

@property(retain, nonatomic) dispatch_queue_t queue;
@property(retain, nonatomic) NSString *name;
@property(nonatomic) struct CGPoint bluePrimary;
@property(nonatomic) struct CGPoint greenPrimary;
@property(nonatomic) struct CGPoint redPrimary;
@property(nonatomic) unsigned int maxPixelsHigh;
@property(nonatomic) unsigned int maxPixelsWide;
@property(nonatomic) struct CGSize sizeInMillimeters;
@property(nonatomic) unsigned int serialNum;
@property(nonatomic) unsigned int productID;
@property(nonatomic) unsigned int vendorID;
- (void)dealloc;
- (id)init;
@property(copy, nonatomic) void (^terminationHandler)(void);
- (id)dispatchQueue;
- (void)setDispatchQueue:(id)arg1;

@end
