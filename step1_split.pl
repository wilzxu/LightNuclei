#!/usr/bin/perl
#
srand($ARGV[0]);

@all_image=glob "./data/train_image_flip/*";


foreach $image (@all_image){
	$r=rand(1);
	if ($r<0.8){
		@t=split '/', $image;
		$train{$t[-1]}=0;
		
	}else{
		$test{$image}=0;
		@t=split '/', $image;
		$est{$t[-1]}=0;
	}
}

open LIST_TEST, ">list_test" or die;
@all=keys %test;
foreach $image (@all){
	print LIST_TEST "$image\n";
}



@splitlist=glob "./data/train_image_flip_iflarge_min200/*";
foreach $file (@splitlist){
	@t=split '/', $file;
	@t=split '_', $t[-1];
	if (exists $test{$t[-1]}){}else{
		$newtrain{$file}=0;
	}
	if ($file=~/TCGA/){
		$newtrain{$file}=0;
	}
	if ($file=~/dna/){
		$newtrain{$file}=0;
	}
	$count{$t[-1]}++;
}
	
@all_keys=sort{$count{$b}<=>$count{$a}}keys %count;
print "$count{$all_keys[0]}\n";
$max=$count{$all_keys[0]};



@all=keys %newtrain;
open LIST_TRAIN_1, ">list_train_1.1" or die;
open LIST_TRAIN_2, ">list_train_1.2" or die;
foreach $image (@all){
	$label=$image;
	$label=~s/train_image_pad/train_label/g;
	@t=split '/', $image;
	@t=split '_', $t[-1];
	$ccc=$count{$t[-1]};
	$ratio=$max/$ccc;
	print "$ratio\n";
	$r=rand(1);
		$iii=0;
		while ($iii<$ratio){
			print LIST_TRAIN_1 "$image\t$label\n";
			$iii++;
		}
		$iii=0;
		while ($iii<$ratio){
			print LIST_TRAIN_2 "$image\t$label\n";
			$iii++;
		}
}
close TRAIN_PATCH;


open LIST_TRAIN_1, ">list_train_2.1" or die;
open LIST_TRAIN_2, ">list_train_2.2" or die;
foreach $image (@all){
	$label=$image;
	$label=~s/train_image_pad/train_label/g;
	@t=split '/', $image;
	@t=split '_', $t[-1];
	$ccc=$count{$t[-1]};
	$ratio=$max/$ccc;
	print "$ratio\n";
	$r=rand(1);
		$iii=0;
		while ($iii<$ratio){
			print LIST_TRAIN_1 "$image\t$label\n";
			$iii++;
		}
		$iii=0;
		while ($iii<$ratio){
			print LIST_TRAIN_2 "$image\t$label\n";
			$iii++;
		}
}
close TRAIN_PATCH;




open LIST_TRAIN_1, ">list_train_3.1" or die;
open LIST_TRAIN_2, ">list_train_3.2" or die;
foreach $image (@all){
	$label=$image;
	$label=~s/train_image_pad/train_label/g;
	@t=split '/', $image;
	@t=split '_', $t[-1];
	$ccc=$count{$t[-1]};
	$ratio=$max/$ccc;
	print "$ratio\n";
	$r=rand(1);
		$iii=0;
		while ($iii<$ratio){
			print LIST_TRAIN_1 "$image\t$label\n";
			$iii++;
		}
		$iii=0;
		while ($iii<$ratio){
			print LIST_TRAIN_2 "$image\t$label\n";
			$iii++;
		}
}
close TRAIN_PATCH;


open LIST_TRAIN_1, ">list_train_4.1" or die;
open LIST_TRAIN_2, ">list_train_4.2" or die;
foreach $image (@all){
	$label=$image;
	$label=~s/train_image_pad/train_label/g;
	@t=split '/', $image;
	@t=split '_', $t[-1];
	$ccc=$count{$t[-1]};
	$ratio=$max/$ccc;
	print "$ratio\n";
	$r=rand(1);
		$iii=0;
		while ($iii<$ratio){
			print LIST_TRAIN_1 "$image\t$label\n";
			$iii++;
		}
		$iii=0;
		while ($iii<$ratio){
			print LIST_TRAIN_2 "$image\t$label\n";
			$iii++;
		}
}
close TRAIN_PATCH;

open LIST_TRAIN_1, ">list_train_5.1" or die;
open LIST_TRAIN_2, ">list_train_5.2" or die;
foreach $image (@all){
	$label=$image;
	$label=~s/train_image_pad/train_label/g;
	@t=split '/', $image;
	@t=split '_', $t[-1];
	$ccc=$count{$t[-1]};
	$ratio=$max/$ccc;
	print "$ratio\n";
	$r=rand(1);
		$iii=0;
		while ($iii<$ratio){
			print LIST_TRAIN_1 "$image\t$label\n";
			$iii++;
		}
		$iii=0;
		while ($iii<$ratio){
			print LIST_TRAIN_2 "$image\t$label\n";
			$iii++;
		}
}
close TRAIN_PATCH;

