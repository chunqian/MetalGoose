#import "CDStructures.h"
#import <Foundation/Foundation.h>

@interface CGVirtualDisplay : NSObject {
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
  void *_client;
  unsigned int _displayID;
  unsigned int _hiDPI;
  NSArray *_modes;
  unsigned int _serverRPC_port;
  unsigned int _proxyRPC_port;
  unsigned int _clientHandler_port;
}

@property(readonly, nonatomic) NSArray *modes;
@property(readonly, nonatomic) unsigned int hiDPI;
@property(readonly, nonatomic) unsigned int displayID;
@property(readonly, nonatomic) CDUnknownBlockType terminationHandler;
@property(readonly, nonatomic) dispatch_queue_t queue;
@property(readonly, nonatomic) struct CGPoint whitePoint;
@property(readonly, nonatomic) struct CGPoint bluePrimary;
@property(readonly, nonatomic) struct CGPoint greenPrimary;
@property(readonly, nonatomic) struct CGPoint redPrimary;
@property(readonly, nonatomic) unsigned int maxPixelsHigh;
@property(readonly, nonatomic) unsigned int maxPixelsWide;
@property(readonly, nonatomic) struct CGSize sizeInMillimeters;
@property(readonly, nonatomic) NSString *name;
@property(readonly, nonatomic) unsigned int serialNum;
@property(readonly, nonatomic) unsigned int productID;
@property(readonly, nonatomic) unsigned int vendorID;
- (BOOL)applySettings:(id)arg1;
- (void)dealloc;
- (id)initWithDescriptor:(id)arg1;

@end
