#!/usr/bin/perl
#
open FILE, "./data/stage1_train_labels.csv" or die;
$line=<FILE>;
chomp $line;
@t=split ",", $line;
$name=$t[0];
mkdir "./data/split_label";
mkdir "./data/train_label";
while ($line=<FILE>){
	chomp $line;
	@table=split ",", $line;
	if ($table[0] eq $name){
		print NEW "$line\n";
	}else{
		$name=$table[0];
		open NEW, ">./data/split_label/$name" or die;
		print NEW "$line\n";
	}
}
close FILE;
